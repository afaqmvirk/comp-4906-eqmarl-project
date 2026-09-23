#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
from _study2_bootstrap import ROOT
import numpy as np
import tensorflow as tf
from eqmarl_network_study.learning_noise.training import build_training_components, seed_everything
from eqmarl_network_study.learning_noise.shot_allocation import compare_allocations


def run(args):
    tf.get_logger().setLevel("ERROR")
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError("Refusing to overwrite %s" % output)
    env, actor, critic, _, _, _ = build_training_components("eqmarl_psi_plus", args.q, 1, args.seed, noise_backend="exact_density")
    if args.weights:
        with np.load(ROOT / args.weights) as weights:
            for name, model in [("actor", actor), ("critic", critic)]:
                model.set_weights([weights["%s_%d"%(name,i)] for i in range(len(model.get_weights()))])
    policy = tf.function(lambda x: actor(x), input_signature=[tf.TensorSpec((None,36),tf.float32)])
    seed_everything(args.seed+200000)
    states, _ = env.reset()
    current, following, rewards, discounts = [], [], [], []
    for index in range(args.transitions):
        probs = policy(tf.convert_to_tensor(states,tf.float32)).numpy()
        actions = [np.random.choice(4, p=row) for row in probs]
        next_states, reward, done, truncated, _ = env.step(actions)
        current.append(states)
        following.append(next_states)
        rewards.append(np.sum(reward))
        discounts.append(.99 * (not any(done)))
        states = next_states
        if any(done) or any(truncated):
            states, _ = env.reset()
    all_states = tf.constant(np.concatenate([current,following]), tf.float32)
    raw_graph = tf.function(lambda x: critic.layers[1](critic.layers[0](x)))
    raw = np.concatenate([raw_graph(all_states[i:i+50]).numpy()[:,0] for i in range(0,len(all_states),50)])
    expectations = np.stack([raw[:args.transitions],raw[args.transitions:]],axis=-1)
    scale = float(critic.get_weights()[-1].ravel()[0])
    report = compare_allocations(expectations, rewards, discounts, scale,
        trials=args.trials, seed=args.seed+300000, allocation_margin=args.allocation_margin)
    report.update(q=args.q, training_seed=args.seed, source_weights=args.weights,
        frozen_model="trained checkpoint" if args.weights else "initialization",
        diagnostic_environment_seed=args.seed+200000)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w",encoding="utf-8") as handle:
        json.dump(report,handle,indent=2,allow_nan=False)
    np.savez_compressed(output.with_suffix(".npz"), expectations=expectations,
        rewards=rewards, discounts=discounts, current_states=current, next_states=following)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights")
    parser.add_argument("--q",type=float,default=.1)
    parser.add_argument("--seed",type=int,default=101)
    parser.add_argument("--transitions",type=int,default=500)
    parser.add_argument("--trials",type=int,default=500)
    parser.add_argument("--allocation-margin",choices=["raw","lower_confidence"],default="raw")
    parser.add_argument("--output",default="results/feasibility/shots_initial.json")
    args = parser.parse_args()
    if args.transitions < 1 or args.trials < 2:
        parser.error("Positive transitions and at least two trials required")
    run(args)
