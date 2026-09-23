#!/usr/bin/env python
import argparse
import json
import os
import random
import time
from pathlib import Path

_early_parser = argparse.ArgumentParser(add_help=False)
_early_parser.add_argument("--threads", type=int, default=2)
_early_args, _ = _early_parser.parse_known_args()
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(_early_args.threads))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", str(_early_args.threads))

from _study2_bootstrap import ROOT
import numpy as np
import tensorflow as tf
import eqmarl
from eqmarl_network_study.learning_noise.training import (
    InstrumentedMAA2C, build_training_components, seed_everything,
    software_versions, _json_safe,
)


class BatchedPolicyMAA2C(InstrumentedMAA2C):
    def policy(self, states, batched=False):
        if batched:
            return super().policy(states, batched=True)
        probabilities = self.policy_graph(tf.convert_to_tensor(states, tf.float32))
        values = probabilities.numpy()
        actions = [np.random.choice(len(row), p=row) for row in values]
        return actions, [row[None,:] for row in probabilities]

    def update(self, batch):
        data = [tf.convert_to_tensor(np.array([getattr(item, key) for item in batch]), dtype)
                for key, dtype in [("states",tf.float32),("actions",tf.int32),
                    ("rewards",tf.float32),("next_states",tf.float32),("dones",tf.float32)]]
        output = self.train_step(*data)
        for label, value in zip(["actor_loss", "critic_loss", "value_mean", "value_std", "actor_gradient_norm", "critic_gradient_norm"], output):
            getattr(self,label+"_history").append(float(value.numpy()))

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        rewards = tf.reduce_sum(rewards, axis=-1, keepdims=True)
        dones = tf.cast(tf.reduce_sum(dones,axis=-1,keepdims=True)>0.,tf.float32)
        with tf.GradientTape() as ct, tf.GradientTape() as at:
            values = self.model_critic(states)
            next_values = self.model_critic(next_states)
            logs = []
            for agent in range(self.n_envs):
                probs = self.model_actor(states[:,agent])
                indices = tf.stack([tf.range(tf.shape(actions)[0]),actions[:,agent]],axis=1)
                logs.append(tf.math.log(tf.gather_nd(probs,indices)))
            logs = tf.stack(logs,axis=1)
            targets = rewards + (1.-dones)*self.gamma*next_values
            advantage = targets-values

            entropy = tf.reduce_mean(-tf.reduce_sum(probs*tf.math.log(probs),axis=-1))
            actor_loss = tf.reduce_mean(-logs*advantage)+self.alpha*entropy
            critic_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)(values,targets)
        ag = at.gradient(actor_loss,self.model_actor.trainable_variables)
        cg = ct.gradient(critic_loss,self.model_critic.trainable_variables)
        for optimizers, grads, variables in [(self.optimizer_actor,ag,self.model_actor.trainable_variables),(self.optimizer_critic,cg,self.model_critic.trainable_variables)]:
            if isinstance(optimizers,(list,tuple)):
                for i,optimizer in enumerate(optimizers):
                    optimizer.apply_gradients([(grads[i],variables[i])])
            else:
                optimizers.apply_gradients(zip(grads,variables))
        return actor_loss,critic_loss,tf.reduce_mean(values),tf.math.reduce_std(values),tf.linalg.global_norm(ag),tf.linalg.global_norm(cg)


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(data), handle, indent=2, allow_nan=False)
    temporary.replace(path)


DIAGNOSTICS = ["actor_loss", "critic_loss", "value_mean", "value_std", "actor_gradient_norm", "critic_gradient_norm"]


def rng_state():
    state = np.random.get_state()
    return dict(numpy=[state[0],state[1].tolist(),state[2],state[3],state[4]], python=random.getstate())


def restore_rng(state):
    ns = state["numpy"]
    np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
    ps = state["python"]
    random.setstate((ps[0],tuple(ps[1]),ps[2]))


def training_checkpoint(actor, critic, ao, co):
    objects = dict(actor=actor,critic=critic)
    objects.update({"actor_optimizer_%d"%i:optimizer for i,optimizer in enumerate(ao)})
    objects.update({"critic_optimizer_%d"%i:optimizer for i,optimizer in enumerate(co)})
    if hasattr(critic.layers[1],"_noise_generator"):
        objects["noise_generator"] = critic.layers[1]._noise_generator
    return tf.train.Checkpoint(**objects)


def run(args):
    tf.get_logger().setLevel("ERROR")
    output = ROOT / args.output_root / args.group
    output.mkdir(parents=True, exist_ok=True)
    name = "%s_q%g_r%d_seed%d" % (args.backend, args.q, args.repetitions, args.seed)
    path = output / (name + ".json")
    if path.exists() and not args.resume:
        raise FileExistsError("Refusing to replace an existing run: %s" % path)
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if previous:
        if previous["status"] == "complete":
            print("Already complete: %s"%path,flush=True)
            return
        if previous["episodes_requested"] != args.episodes:
            raise ValueError("Resume requires the original episode horizon")
        if "checkpoint" not in previous:
            raise ValueError("Run has no resumable checkpoint")
    start = time.perf_counter()
    env, actor, critic, ao, co, _ = build_training_components(
        "eqmarl_psi_plus", args.q, args.repetitions, args.seed,
        noise_bank_size=1024, noise_backend=args.backend)

    original_actor_call, original_critic_call = actor.call, critic.call
    actor_graph = tf.function(lambda x: original_actor_call(x), input_signature=[tf.TensorSpec((None,36),tf.float32)])
    critic_graph = tf.function(lambda x: original_critic_call(x), input_signature=[tf.TensorSpec((None,2,36),tf.float32)])
    if args.backend == "exact_density":
        critic.call = lambda inputs, training=None, mask=None: critic_graph(tf.cast(inputs,tf.float32))
    algorithm = BatchedPolicyMAA2C(env=env, model_actor=actor, model_critic=critic,
        optimizer_actor=ao, optimizer_critic=co, gamma=.99, alpha=.001,
        episode_metrics_callback=eqmarl.environments.coin_game.episode_metrics_callback)
    algorithm.policy_graph = actor_graph
    checkpoint = training_checkpoint(actor,critic,ao,co)
    checkpoint_directory = output/(name+"_checkpoints")
    checkpoint_directory.mkdir(exist_ok=True)
    seed_everything(args.seed)
    result = dict(status="running", framework="eqmarl_psi_plus", backend=args.backend,
        q=args.q, seed=args.seed, repetitions=args.repetitions,
        episodes_requested=args.episodes, episodes_completed=0, fresh_training=True,
        measurement_shots=None, gradient_estimator="exact autodiff" if args.backend=="exact_density" else "Pauli trajectory adjoint",
        purpose=args.purpose,
        update="upstream MAA2C semantics, compiled model calls, batched actor inference",
        versions=software_versions(), trajectory={},
        environment="CoinGame-2 fully observed MDP", steps_per_episode=50,
        agents=2, qubits_per_agent=4, critic_vqc_blocks=5, gamma=.99,
        entropy_coefficient=.001, evaluation_episodes_requested=args.eval_episodes,
        execution_threads=args.threads)
    first_episode, elapsed_before, metrics, restoration = 0, 0., [], None
    if previous:
        result = previous
        first_episode = result["episodes_completed"]
        elapsed_before = result["runtime_seconds"]
        restoration = checkpoint.read(str(ROOT/result["checkpoint"]))
        restoration.assert_existing_objects_matched()
        restore_rng(result["rng_state"])
        for label in DIAGNOSTICS:
            setattr(algorithm,label+"_history",list(result["trajectory"][label]))
        metric_keys = [key for key in result["trajectory"] if key not in DIAGNOSTICS]
        metrics = [{key:result["trajectory"][key][i] for key in metric_keys} for i in range(first_episode)]
    result["status"] = "running"
    write_json(path, result)
    last_episode = min(args.episodes,first_episode+args.chunk_episodes)
    for episode in range(first_episode,last_episode):
        _, _, _ = algorithm.run_episode(episode, episode*50, 50)
        if restoration is not None:
            restoration.assert_consumed()
            restoration = None
        metrics.append(algorithm.episode_metrics_callback(env))
        if (episode+1) % args.log_every == 0 or episode+1 == last_episode:
            trajectory = {key:[row[key] for row in metrics] for key in metrics[0]}
            for label in ["actor_loss", "critic_loss", "value_mean", "value_std", "actor_gradient_norm", "critic_gradient_norm"]:
                trajectory[label] = getattr(algorithm,label+"_history")
            if not all(np.isfinite(value).all() for value in trajectory.values()):
                result["status"] = "failed_nonfinite"
                write_json(path, result)
                raise RuntimeError("Non-finite trajectory")
            prefix = checkpoint.write(str(checkpoint_directory/("episode_%06d"%(episode+1))))
            result.update(episodes_completed=episode+1, runtime_seconds=elapsed_before+time.perf_counter()-start,
                trajectory=trajectory,checkpoint=str(Path(prefix).relative_to(ROOT)),rng_state=rng_state())
            write_json(path, result)
            print(json.dumps(dict(run=name, episode=episode+1,
                final100=float(np.mean(trajectory["undiscounted_reward"][-100:])),
                elapsed_s=round(result["runtime_seconds"],1))), flush=True)
    if last_episode < args.episodes:
        result["status"] = "paused_checkpoint"
        write_json(path,result)
        return

    seed_everything(args.seed + 100000)
    evaluation = []
    for episode in range(args.eval_episodes):
        states, _ = env.reset()
        for step in range(50):
            actions, _ = algorithm.policy(states)
            states, _, dones, truncated, _ = env.step(actions)
            if any(dones) or any(truncated):
                break
        evaluation.append(algorithm.episode_metrics_callback(env)["undiscounted_reward"])
    result.update(status="complete", evaluation_scores=evaluation,
        evaluation_mean=float(np.mean(evaluation)) if evaluation else None,
        runtime_seconds=elapsed_before+time.perf_counter()-start)

    weights = {"actor_%d"%i:w for i,w in enumerate(actor.get_weights())}
    weights.update({"critic_%d"%i:w for i,w in enumerate(critic.get_weights())})
    np.savez_compressed(output/(name+"_weights.npz"), **weights)
    write_json(path, result)

    print(json.dumps({key:value for key,value in result.items() if key not in ["trajectory", "evaluation_scores", "versions", "rng_state"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["exact_density", "pauli_trajectory_adjoint"], default="exact_density")
    parser.add_argument("--q", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--group", default="precision_v2")
    parser.add_argument("--output-root",default="results/feasibility")
    parser.add_argument("--purpose",default="simulation-precision feasibility, not hardware error mitigation")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--chunk-episodes",type=int,default=500)
    args = parser.parse_args()
    if args.episodes < 1 or args.eval_episodes < 0 or args.log_every < 1 or args.chunk_episodes < 1:
        parser.error("Invalid episode/log counts")
    try:
        run(args)
    except Exception as exc:
        failure_path = ROOT / args.output_root / args.group / ("%s_q%g_r%d_seed%d.json" % (args.backend,args.q,args.repetitions,args.seed))
        if failure_path.exists() and not isinstance(exc, FileExistsError):
            with failure_path.open(encoding="utf-8") as handle:
                failed = json.load(handle)
            failed.update(status="failed", exception=repr(exc))
            write_json(failure_path, failed)
        raise
