from copy import deepcopy
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from eqmarl_network_study.learning_noise.exact_study import (
    COMMIT, atomic_json, comparable_metrics, inventory, make_manifest, manifest_matches, paired_summary,
    read_json, sha256, treatment_order, validate_record,
)

ROOT = Path(__file__).resolve().parents[1]


def record(framework="eqmarl_psi_plus",q=.1,seed=101):
    values = np.concatenate([np.full(100,10.),np.full(2900,25.)]).tolist()
    return dict(status="complete",framework=framework,backend="exact_density",q=q,seed=seed,
                fresh_training=True,versions={"eqmarl_upstream_commit":COMMIT},episodes_requested=3000,
                episodes_completed=3000,trajectory={"undiscounted_reward":values},evaluation_scores=[24.]*100)


def import_script(name):
    sys.path.insert(0,str(ROOT/"scripts"))
    spec = importlib.util.spec_from_file_location(name,ROOT/"scripts"/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_grid_and_reuse_are_exact():
    manifest = make_manifest(ROOT,read_json(ROOT/"configs/study2_exact.json"))
    origins = [row["origin"] for row in manifest["records"]]
    assert len(origins)==28
    assert origins.count("existing_exact")==3
    assert origins.count("new_exact")==20
    assert origins.count("existing_classical")==5
    assert len(manifest["code_sha256"])>=7
    rows = inventory(ROOT,manifest)
    assert all(row["status"]=="complete" for row in rows if row["origin"]!="new_exact")


def test_cleanup_accepts_known_edits_but_rejects_new_code_or_protocol():
    recorded = read_json(ROOT/"results/study2_exact/manifest.json")
    current = make_manifest(ROOT, recorded["config"])
    assert manifest_matches(ROOT, current, recorded)
    changed = deepcopy(current)
    changed["code_sha256"][next(iter(changed["code_sha256"]))] = "0"*64
    assert not manifest_matches(ROOT, changed, recorded)
    changed = deepcopy(current)
    changed["config"]["episodes"] = 100
    assert not manifest_matches(ROOT, changed, recorded)


@pytest.mark.parametrize("change",[
    {"backend":"pauli_trajectory_adjoint"},{"q":.03},{"seed":202},
    {"episodes_requested":2999},{"fresh_training":False},{"status":"paused_checkpoint"},
    {"evaluation_scores":[float("nan")]*100},{"episodes_completed":3001},
    {"steps_per_episode":49},{"versions":{}},
])
def test_invalid_results_rejected(change):
    value = record()
    value.update(change)
    with pytest.raises(ValueError):
        validate_record(value,"eqmarl_psi_plus",.1,101)


def test_inventory_pins_reused_inputs(tmp_path):
    path = tmp_path/"classical.json"
    atomic_json(path,record("sctde",0.))
    entry = dict(id="classical",framework="sctde",q=0.,seed=101,path="classical.json",source_sha256=sha256(path))
    manifest = dict(config={"episodes":3000},records=[entry])
    assert inventory(tmp_path,manifest)[0]["status"]=="complete"
    value = read_json(path)
    value["trajectory"]["undiscounted_reward"][0] += 1
    atomic_json(path,value)
    assert inventory(tmp_path,manifest)[0]["status"]=="invalid"


def test_rolling_window_one_based_and_censoring():
    value = record()
    metrics = comparable_metrics(value,100)
    assert metrics["crossing_20"]==167
    assert metrics["crossing_25"]==200
    assert comparable_metrics(value,10)["crossing_20"]==107
    value["trajectory"]["undiscounted_reward"] = [19.]*3000
    metrics = comparable_metrics(value,100)
    assert metrics["crossing_20"] is None
    assert metrics["restricted_20"]==3000 and not metrics["reached_20"]
    value["trajectory"]["undiscounted_reward"][-1] = 119.
    metrics = comparable_metrics(value,100)
    assert metrics["crossing_20"]==3000
    assert metrics["restricted_20"]==3000 and metrics["reached_20"]


def test_verified_matched_result_stays_546_748():
    for q,expected in [(0.,546),(.1,748)]:
        value = read_json(ROOT/("results/feasibility/precision_v2/exact_density_q%g_r1_seed101.json"%q))
        assert comparable_metrics(value,100)["crossing_20"]==expected


def test_paired_statistics_do_not_treat_episodes_as_replicates():
    paired = paired_summary({seed:2*seed for seed in range(1,6)},{seed:seed for seed in range(1,6)})
    assert paired["n"]==5
    assert paired["mean_left_minus_right"]==3.
    assert paired["positive"]==5
    assert paired["exact_sign_flip_p_two_sided"]==.0625
    one = paired_summary({1:3},{1:1})
    assert one["bootstrap_ci95"] is None and one["exact_sign_flip_p_two_sided"] is None


def test_controller_uses_exact_backend_and_new_root():
    module = import_script("run_study2_exact")
    config = read_json(ROOT/"configs/study2_exact.json")
    row = dict(q=.03,seed=303)
    args = module.command(row,config)
    assert args[args.index("--backend")+1]=="exact_density"
    assert args[args.index("--output-root")+1]=="results/study2_exact"
    assert args[args.index("--episodes")+1]=="3000"
    assert "--resume" in args
    assert treatment_order(dict(q=.1,seed=505),config)<treatment_order(dict(q=.01,seed=101),config)


def test_complete_outputs_and_incomplete_rejection(tmp_path,monkeypatch):
    module = import_script("analyze_study2_exact")
    config = read_json(ROOT/"configs/study2_exact.json")
    config["output_root"]="output"

    manifest = make_manifest(ROOT,read_json(ROOT/"configs/study2_exact.json"))
    manifest = deepcopy(manifest)
    manifest["config"] = config
    for i,entry in enumerate(manifest["records"]):
        entry["path"]="input%d.json"%i
        entry.pop("source_sha256",None)
        value = record(entry["framework"],entry["q"],entry["seed"])
        if entry["framework"]=="sctde":
            value["trajectory"]["undiscounted_reward"] = [10.]*200+[25.]*2800
        atomic_json(tmp_path/entry["path"],value)
        (tmp_path/entry["path"]).with_name("input%d_weights.npz"%i).touch()
    monkeypatch.setattr(module,"make_manifest",lambda root,config:manifest)
    result,runs = module.analyze(tmp_path,manifest)
    assert result["status"]=="complete"
    primary = result["windows"]["100"]
    assert len(primary["runs"])==28
    assert all(row["descriptive_retained_fraction"]==1. for row in primary["retention"])
    module.write_outputs(tmp_path,manifest,result,runs)
    assert not list(tmp_path.rglob("*.md"))
    assert (tmp_path/"output/learning_comparison.png").stat().st_size>1000
    assert read_json(tmp_path/"output/summary.json") == result
    entry = manifest["records"][0]
    value = read_json(tmp_path/entry["path"])
    value["status"]="paused_checkpoint"
    atomic_json(tmp_path/entry["path"],value)
    with pytest.raises(ValueError,match="incomplete/invalid"):
        module.analyze(tmp_path,manifest)


def test_controller_resumes_failed_chunk_then_analyzes(tmp_path,monkeypatch):
    module = import_script("run_study2_exact")
    config = read_json(ROOT/"configs/study2_exact.json")
    config["output_root"]="output"
    output = tmp_path/"output"
    output.mkdir()
    entry = dict(id="test_quantum",framework="eqmarl_psi_plus",q=.1,seed=101,
                 origin="new_exact",path="output/raw/test.json")
    manifest = dict(config=config,records=[entry])
    monkeypatch.setattr(module,"ROOT",tmp_path)
    monkeypatch.setattr(module,"controller_lock",lambda output:nullcontext())
    monkeypatch.setattr(module,"prepare",lambda config:(output,manifest))
    monkeypatch.setattr(module.time,"sleep",lambda seconds:None)
    calls,analyzed = [],[]

    class Child:
        def __init__(self,args,**kwargs):
            calls.append(args)
            self.pid = 100+len(calls)
            value = record()
            value["episodes_completed"] = 100 if len(calls)==1 else (500 if len(calls)==2 else 3000)
            value["status"] = "failed" if len(calls)==1 else ("paused_checkpoint" if len(calls)==2 else "complete")
            value["checkpoint"]="output/checkpoint"
            value["rng_state"]={}
            (output/"checkpoint.index").touch()
            (output/"checkpoint.data-00000-of-00001").touch()
            atomic_json(tmp_path/entry["path"],value)
            (tmp_path/entry["path"]).with_name("test_weights.npz").touch()

        def poll(self):
            return 1 if self.pid==101 else 0

    monkeypatch.setattr(module.subprocess,"Popen",Child)
    monkeypatch.setattr(module.subprocess,"run",lambda args,**kwargs:analyzed.append(args))
    module.run(config)
    assert len(calls)==3 and len(analyzed)==1
    progress = read_json(output/"progress.json")
    assert progress["status"]=="complete"
    assert progress["failures"]=={"test_quantum":1}
    assert progress["quantum_complete"]==1
