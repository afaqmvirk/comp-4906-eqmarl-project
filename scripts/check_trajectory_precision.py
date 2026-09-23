#!/usr/bin/env python
import argparse
import json
import os
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
from _study2_bootstrap import ROOT
import numpy as np
import tensorflow as tf
from eqmarl_network_study.learning_noise.training import build_training_components


def flatten_gradient(gradients):
    return tf.concat([tf.reshape(item,[-1]) for item in gradients],0)


def run(args):
    tf.get_logger().setLevel("ERROR")
    output = ROOT/"results"/"feasibility"/"trajectory_precision_initial.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite %s"%output)
    with np.load(ROOT/"results"/"feasibility"/"shots_initial.npz") as data:
        states = tf.constant(data["current_states"][:8],tf.float32)
    _,_,exact,_,_,_ = build_training_components("eqmarl_psi_plus",.1,1,101,noise_backend="exact_density")
    with tf.GradientTape() as tape:
        ref_values = exact(states)
        linear_objective = tf.reduce_mean(ref_values)
    ref_grad = flatten_gradient(tape.gradient(linear_objective,exact.trainable_variables)).numpy()
    ref_values = ref_values.numpy()
    _,_,mc,_,_,_ = build_training_components("eqmarl_psi_plus",.1,1,101,noise_backend="pauli_trajectory_adjoint",noise_bank_size=1024)
    mc.set_weights(exact.get_weights())
    norm = np.linalg.norm(ref_grad)
    rows = []
    for repetitions in [1,8,32]:
        mc.layers[1].noise_repetitions = repetitions
        @tf.function
        def probe(x):
            with tf.GradientTape() as tape:
                values = mc(x)
                objective = tf.reduce_mean(values)
            return values,flatten_gradient(tape.gradient(objective,mc.trainable_variables))
        values, gradients = [], []
        for trial in range(args.trials):
            value, gradient = probe(states)
            values.append(value.numpy())
            gradients.append(gradient.numpy())
        values, gradients = np.asarray(values),np.asarray(gradients)

        row = dict(repetitions=repetitions,trials=args.trials,
            value_rmse=float(np.sqrt(np.mean((values-ref_values)**2))),
            value_mean_bias=float(np.mean(values-ref_values)))
        for name,reference,samples in [("all",ref_grad,gradients),("circuit",ref_grad[:-1],gradients[:,:-1])]:
            reference_norm = np.linalg.norm(reference)
            row[name+"_relative_gradient_rms_error"] = float(np.sqrt(np.mean(np.sum((samples-reference)**2,axis=1)))/reference_norm)
            row[name+"_mean_gradient_relative_error"] = float(np.linalg.norm(samples.mean(axis=0)-reference)/reference_norm)
            cosines = (samples@reference)/(np.maximum(np.linalg.norm(samples,axis=1),1e-30)*reference_norm)
            row[name+"_mean_gradient_cosine"] = float(cosines.mean())
        rows.append(row)
        print(json.dumps(row),flush=True)
    result = dict(scope="frozen initialization, eight observed states, no training",
        q=.1,seed=101,bank_size=1024,objective="mean critic value (linear, not nonlinear RL loss)",
        reference="exact depolarizing channel expectation and gradient",
        exact_gradient_norm=float(norm),exact_circuit_gradient_norm=float(np.linalg.norm(ref_grad[:-1])),
        results=rows,limitations=["A finite bank can retain bias even with more trajectories.",
            "The mean of a nonlinear loss gradient need not equal the loss gradient at the mean value.",
            "Pauli trajectories are not hardware measurement shots."])
    with output.open("w",encoding="utf-8") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",type=int,default=32)
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("At least two trials required")
    run(args)
