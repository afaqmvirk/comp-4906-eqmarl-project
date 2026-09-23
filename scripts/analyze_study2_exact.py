#!/usr/bin/env python
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from _study2_bootstrap import ROOT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eqmarl_network_study.learning_noise.exact_study import (
    atomic_json, comparable_metrics, inventory, make_manifest, manifest_matches, paired_summary, read_json, sha256,
)


def analyze(root,manifest):
    root = Path(root)
    config = manifest["config"]
    if not manifest_matches(root, make_manifest(root,config), manifest):
        raise ValueError("Protocol or source hashes changed")
    coverage = inventory(root,manifest)
    if any(row["status"]!="complete" for row in coverage):
        raise ValueError("Refusing a full-study analysis with incomplete/invalid treatments")
    runs = [(entry,read_json(root/entry["path"])) for entry in manifest["records"]]
    result = dict(status="complete",created_utc=datetime.now(timezone.utc).isoformat(),
                  protocol=config["name"],horizon=config["episodes"],
                  source_sha256={entry["path"]:sha256(root/entry["path"]) for entry,record in runs},
                  code_sha256=manifest["code_sha256"],windows={})
    for window in [config["primary_window"],config["sensitivity_window"]]:
        rows = []
        grouped = defaultdict(list)
        for entry,record in runs:
            metrics = comparable_metrics(record,window,config["episodes"])
            row = dict(id=entry["id"],framework=entry["framework"],q=entry["q"],seed=entry["seed"],
                       origin=entry["origin"],**metrics)
            rows.append(row)
            grouped[(entry["framework"],entry["q"])].append(row)
        groups = []
        for (framework,q),items in sorted(grouped.items()):
            group = dict(framework=framework,q=q,n=len(items),seeds=sorted(row["seed"] for row in items),
                         exploratory=len(items)<5,
                         mean_final100=float(np.mean([row["final100"] for row in items])),
                         mean_training_reward=float(np.mean([row["mean_reward"] for row in items])))
            for target in config["targets"]:
                group["reached_%d"%target] = sum(row["reached_%d"%target] for row in items)
                group["mean_restricted_%d"%target] = float(np.mean([row["restricted_%d"%target] for row in items]))
            groups.append(group)
        comparisons = []
        clean = grouped[("eqmarl_psi_plus",0.)]
        classical = grouped[("sctde",0.)]
        for (framework,q),items in sorted(grouped.items()):
            if framework!="eqmarl_psi_plus":
                continue
            for reference,label in [(clean,"minus_clean_quantum"),(classical,"minus_classical")]:
                if q==0 and label=="minus_clean_quantum":
                    continue
                for metric in ["restricted_20","restricted_25","final100","mean_reward"]:
                    comparison = paired_summary({row["seed"]:row[metric] for row in items},
                                                {row["seed"]:row[metric] for row in reference})
                    comparisons.append(dict(q=q,comparison=label,metric=metric,**comparison))
        retention = []
        for target in config["targets"]:
            metric = "restricted_%d"%target
            classical_by_seed = {row["seed"]:row[metric] for row in classical}
            clean_by_seed = {row["seed"]:row[metric] for row in clean}
            for (framework,q),items in sorted(grouped.items()):
                if framework!="eqmarl_psi_plus" or len(items)!=5:
                    continue
                seeds = sorted(row["seed"] for row in items)
                if seeds!=sorted(clean_by_seed) or seeds!=sorted(classical_by_seed):
                    raise ValueError("Incomplete primary paired group")
                noisy_by_seed = {row["seed"]:row[metric] for row in items}
                clean_advantage = float(np.mean([classical_by_seed[s]-clean_by_seed[s] for s in seeds]))
                noisy_advantage = float(np.mean([classical_by_seed[s]-noisy_by_seed[s] for s in seeds]))
                retention.append(dict(q=q,target=target,clean_advantage_episodes=clean_advantage,
                                      noisy_advantage_episodes=noisy_advantage,
                                      descriptive_retained_fraction=noisy_advantage/clean_advantage if clean_advantage>0 else None))
        result["windows"][str(window)] = dict(runs=rows,groups=groups,comparisons=comparisons,retention=retention)
    result["quantum_only_evaluation"] = [dict(id=entry["id"],mean=float(np.mean(record["evaluation_scores"])),n=100)
                                            for entry,record in runs if entry["framework"]=="eqmarl_psi_plus"]
    return result,runs


def write_outputs(root,manifest,result,runs):
    output = Path(root)/manifest["config"]["output_root"]
    output.mkdir(parents=True,exist_ok=True)
    primary = manifest["config"]["primary_window"]

    fig,axes = plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    classical = [pd.Series(record["trajectory"]["undiscounted_reward"]).rolling(primary).mean().to_numpy()
                 for entry,record in runs if entry["framework"]=="sctde"]
    episodes = np.arange(1,manifest["config"]["episodes"]+1)
    for ax,q,color in zip(axes.flat,[0.,.01,.03,.1],["#2563eb","#08916c","#b16b06","#b43871"]):
        quantum = [pd.Series(record["trajectory"]["undiscounted_reward"]).rolling(primary).mean().to_numpy()
                   for entry,record in runs if entry["framework"]=="eqmarl_psi_plus" and entry["q"]==q]
        for values in classical:
            ax.plot(episodes,values,color="0.5",alpha=.18,lw=.65)
        for values in quantum:
            ax.plot(episodes,values,color=color,alpha=.22,lw=.65)
        ax.plot(episodes,np.mean(classical,axis=0),color="0.25",lw=2,label="Classical SCTDE (5 seeds)")
        ax.plot(episodes,np.mean(quantum,axis=0),color=color,lw=2,label="Exact eQMARL (5 seeds)")
        ax.axhline(20,color="0.6",ls=":",lw=1)
        ax.set_title("Depolarizing probability q = %g"%q)
        ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8,loc="lower right")
    for ax in axes[1]:
        ax.set_xlabel("Training episode")
    for ax in axes[:,0]:
        ax.set_ylabel("Trailing %d-episode team reward"%primary)
    fig.suptitle("Exact-noise Study 2: matched 3,000-episode training trajectories",fontsize=13)
    fig.tight_layout()
    fig.savefig(output/"learning_comparison.png",dpi=180)
    plt.close(fig)
    atomic_json(output/"summary.json",result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",default="results/study2_exact/manifest.json")
    args = parser.parse_args()
    manifest = read_json(ROOT/args.manifest)
    result,runs = analyze(ROOT,manifest)
    write_outputs(ROOT,manifest,result,runs)
    print("Analysis saved to " + manifest["config"]["output_root"],flush=True)
