import sys
from pathlib import Path
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytest.importorskip("tensorflow_quantum")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_noise_feasibility import BatchedPolicyMAA2C, training_checkpoint, rng_state, restore_rng
from eqmarl_network_study.learning_noise.training import InstrumentedMAA2C, build_training_components
from eqmarl.algorithms.algorithm import VectorInteraction
import eqmarl


def test_compiled_update_matches_reference():
    algorithms = []
    for cls in [InstrumentedMAA2C, BatchedPolicyMAA2C]:
        env, actor, critic, ao, co, _ = build_training_components("eqmarl_psi_plus", .1, 1, 601, noise_backend="exact_density")
        algorithms.append(cls(env=env, model_actor=actor, model_critic=critic,
            optimizer_actor=ao, optimizer_critic=co, gamma=.99, alpha=.001,
            episode_metrics_callback=eqmarl.environments.coin_game.episode_metrics_callback))
    for attr in ["model_actor", "model_critic"]:
        getattr(algorithms[1], attr).set_weights(getattr(algorithms[0], attr).get_weights())
    rng = np.random.default_rng(779)
    batch = [VectorInteraction(states=rng.normal(size=(2,36)).astype("float32"),
        actions=rng.integers(0,4,2), action_probs=None, rewards=rng.normal(size=2),
        next_states=rng.normal(size=(2,36)).astype("float32"), dones=[False,i==4]) for i in range(5)]
    for step in range(2):
        for algorithm in algorithms:
            algorithm.update(batch)
        for label in ["actor_loss", "critic_loss", "value_mean", "value_std", "actor_gradient_norm", "critic_gradient_norm"]:
            np.testing.assert_allclose(getattr(algorithms[0], label+"_history")[-1], getattr(algorithms[1], label+"_history")[-1], atol=2e-5, rtol=2e-4)
        for attr in ["model_actor", "model_critic"]:
            for index, (left, right) in enumerate(zip(getattr(algorithms[0],attr).get_weights(), getattr(algorithms[1],attr).get_weights())):
                mask = np.ones(left.shape, dtype=bool)
                if index == 0:

                    if attr == "model_actor":
                        mask[0,:,0] = False
                        mask[-1,:,2] = False
                    else:
                        mask[:,-1,:,2] = False
                np.testing.assert_allclose(left[mask],right[mask],atol=3e-5,rtol=2e-4)
            probe = tf.constant(rng.normal(size=(4,2,36)), tf.float32)
            if attr == "model_actor":
                probe = probe[:,0]
            np.testing.assert_allclose(getattr(algorithms[0],attr)(probe).numpy(),getattr(algorithms[1],attr)(probe).numpy(),atol=3e-5,rtol=2e-4)


def test_batched_actor_inference_matches_individual_calls():
    _, actor, _, _, _, _ = build_training_components("eqmarl_psi_plus", .1, 1, 602, noise_backend="exact_density")
    states = tf.constant(np.random.default_rng(43).normal(size=(2,36)), tf.float32)
    graph = tf.function(lambda x: actor(x))
    batch = graph(states).numpy()
    individual = np.concatenate([actor(states[i:i+1]).numpy() for i in range(2)])
    np.testing.assert_allclose(batch, individual, atol=1e-6)


def test_rng_snapshot_restores_both_action_environment_streams():
    import random
    np.random.seed(37)
    random.seed(92)
    state = rng_state()
    expected = [np.random.random(),random.random()]
    restore_rng(state)
    assert expected == [np.random.random(),random.random()]


@pytest.mark.parametrize("backend", ["exact_density", "pauli_trajectory_adjoint"])
def test_checkpoint_roundtrip_restores_weights_and_optimizer_slots(tmp_path, backend):
    env, actor, critic, ao, co, _ = build_training_components("eqmarl_psi_plus",.1,1,301,noise_bank_size=32,noise_backend=backend)
    algorithm = BatchedPolicyMAA2C(env=env,model_actor=actor,model_critic=critic,
        optimizer_actor=ao,optimizer_critic=co,gamma=.99,alpha=.001,
        episode_metrics_callback=eqmarl.environments.coin_game.episode_metrics_callback)
    rng = np.random.default_rng(61)
    batch = [VectorInteraction(states=rng.normal(size=(2,36)).astype("float32"),
        actions=[0,1],action_probs=None,rewards=[0,1],
        next_states=rng.normal(size=(2,36)).astype("float32"),dones=[False,False])]
    algorithm.update(batch)
    checkpoint = training_checkpoint(actor,critic,ao,co)
    prefix = checkpoint.write(str(tmp_path/"checkpoint"))
    algorithm.update(batch)
    expected = actor.get_weights()+critic.get_weights()
    expected_optimizer = [[variable.numpy().copy() for variable in optimizer.variables()] for optimizer in ao+co]
    checkpoint.read(prefix).assert_consumed()
    algorithm.update(batch)
    for a,b in zip(expected,actor.get_weights()+critic.get_weights()):
        np.testing.assert_array_equal(a,b)
    for expected_variables,optimizer in zip(expected_optimizer,ao+co):
        for a,b in zip(expected_variables,optimizer.variables()):
            np.testing.assert_array_equal(a,b.numpy())
