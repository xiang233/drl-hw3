from logging import info
from typing import Optional, Sequence
import numpy as np
import torch

from src.policies import MLPPolicyPG
from src.critics import ValueCritic
import src.pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        obs = np.concatenate(obs)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        terminals = np.concatenate(terminals)
        q_values = np.concatenate(q_values)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # update the PG actor/policy network once using the advantages
        info: dict = self.actor.update(obs, actions, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            critic_info: dict = {}
            for _ in range(self.baseline_gradient_steps):
                critic_info = self.critic.update(obs, q_values)
            info.update(critic_info)

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # TODO: use the helper function self._discounted_return to calculate the Q-values
            q_values = None
            ############################
            # YOUR IMPLEMENTATION HERE #
            out = []
            for r in rewards:
                out.append(np.array(self._discounted_return(r), dtype=np.float32))
            q_values = out
            ############################

        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
            q_values = None

            ############################
            # YOUR IMPLEMENTATION HERE #
            out = []
            for r in rewards:
                out.append(np.array(self._discounted_reward_to_go(r), dtype=np.float32))
            q_values = out
            ############################

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        # if self.critic is None:   
        #     advantages = q_values.copy()
        # else:
        #     values = self.critic.forward(ptu.from_numpy(obs)).detach().cpu().numpy().squeeze()
        #     advantages = q_values - values
        #     assert values.shape == q_values.shape

        # # normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        # if self.normalize_advantages:
        #     adv_mean = advantages.mean()
        #     adv_std = advantages.std()
        #     if adv_std == 0:
        #         advantages = advantages - adv_mean
        #     else:
        #         advantages = (advantages - adv_mean) / (adv_std + 1e-8)


        advantages = q_values.copy()

    
        if self.critic is not None:
            with torch.no_grad():
                vpred = self.critic(ptu.from_numpy(obs))  # (N,)
            vpred = ptu.to_numpy(vpred)
            assert vpred.shape == q_values.shape
            advantages = q_values - vpred

        if self.normalize_advantages:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!

        self.gamma
        """

        ############################
        # YOUR IMPLEMENTATION HERE #
        disc = 1.0
        total = 0.0
        for r in rewards:
            total += disc * r
            disc *= self.gamma
        out = [float(total)] * len(rewards)
        return out
    
        ############################
        pass


    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.

        self.gamma
        """

        ############################
        # YOUR IMPLEMENTATION HERE #
        T = len(rewards)
        out = [0.0] * T
        running = 0.0
        for i in range(T - 1, -1, -1):
            running = rewards[i] + self.gamma * running
            out[i] = float(running)
        return out
        ############################
        pass
