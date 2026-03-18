import random

import numpy as np
import pytest
import torch
import gymnasium as gym

from ddpg.ddpg import DDPG
from ddpg.actor import ActorMLP
from ddpg.critic import CriticMLP


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_policy(agent: DDPG, env_id: str, n_episodes: int = 5, seed: int = 123) -> float:
    env = gym.make(env_id)
    returns = []

    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0

        while not done:
            a = agent.get_action(s, noise=False)
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_return += r

        returns.append(ep_return)

    env.close()
    return float(np.mean(returns))


class TestDDPGLearning:
    def test_ddpg_learns_on_pendulum(self) -> None:
        seed = 0 
        set_seeds(seed)
        env = gym.make("Pendulum-v1")

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_scale = float(env.action_space.high[0])

        actor = ActorMLP(state_dim, 64, 64, action_dim, action_scale)
        critic = CriticMLP(state_dim, 64, 64, action_dim)

        agent = DDPG(
            actor=actor,
            critic=critic,
            lr_actor=1e-3,
            lr_critic=1e-3,
            timesteps=5000,
            gamma=0.99,
            tau=0.005,
            batch_size=64,
            device="cpu",
            noise_std=0.1,
            buffer_capacity=50_000,
            buffer_start_size=1000,
            n_eval_runs=3,
            eval_every=10_000,   # disable eval spam during train
            save_every=10_000,   # disable frequent checkpointing
            seed=seed,
        )

        before = evaluate_policy(agent, "Pendulum-v1", n_episodes=5, seed=100)
        agent.train(env)
        after = evaluate_policy(agent, "Pendulum-v1", n_episodes=5, seed=200)

        print(f"\nAverage return before training: {before:.2f}")
        print(f"Average return after training:  {after:.2f}")

        assert after > before + 100.0