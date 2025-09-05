import collections
import flax.linen as nn
import jax
import jax.numpy as jnp

Trajectory = collections.namedtuple('Trajectory', 'obs, val, action, log_prob, reward, done')

class PPO:
    _clip_epsilon: float
    _entropy_coefficient: float
    _gamma: float
    _lambda: float
    _parallel_envs: int

    def __init__(
            self,
            clip_epsilon: float,
            entropy_coefficient: float,
            gamma: float,
            lambda_val: float,
            parallel_envs: int) -> None:
        self._clip_epsilon = clip_epsilon
        self._entropy_coefficient = entropy_coefficient
        self._gamma = gamma
        self._lambda = lambda_val
        self._parallel_envs = parallel_envs


    def calc_advantage_delta(self, values: jnp.ndarray, n_values: jnp.ndarray, rewards: jnp.ndarray, dones: jnp.ndarray, tau: float) -> jnp.ndarray:
        return rewards + self._gamma * n_values * (1 - dones) - ((1 + tau) * values)


    def calc_advantages(self, values: jnp.ndarray, rewards: jnp.ndarray, dones: jnp.ndarray, final_values: jnp.ndarray, tau: float) -> jnp.ndarray:
        n_values = values[1:]
        n_values = jnp.append(n_values, jnp.expand_dims(final_values, 0), axis=0)

        deltas = self.calc_advantage_delta(values, n_values, rewards, dones, tau)

        def scan_fn(carry, x):
            new_carry =  x[0] + self._gamma * self._lambda * carry * (1 - x[1])
            return new_carry, new_carry
        
        _, advantages = jax.lax.scan(scan_fn, jnp.zeros(self._parallel_envs), (deltas, dones,), reverse=True)

        return advantages


    def calc_returns(self, values: jnp.ndarray, advantages: jnp.ndarray) -> jnp.ndarray:
        return values + advantages


    def value_loss(self, y_hat: jnp.ndarray, y: jnp.ndarray, value_clip_epsilon: float = float('inf')) -> jnp.ndarray:
        return jnp.clip(jnp.mean(jnp.pow(y - y_hat, 2)), -value_clip_epsilon, value_clip_epsilon)
        

    def policy_loss(self, logits: jnp.ndarray, old_log_probs: jnp.ndarray, advantages: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        one_hot_actions = jax.nn.one_hot(actions, num_classes=logits.shape[-1])

        log_probs = jax.nn.log_softmax(logits)
        log_probs = jnp.sum(log_probs * one_hot_actions, axis=-1)

        ratio = jnp.exp(log_probs - old_log_probs)

        loss = jnp.minimum(ratio * advantages, jnp.clip(ratio, 1 - self._clip_epsilon, 1 + self._clip_epsilon) * advantages)
        loss = -jnp.mean(loss)

        return loss


    def entropy_bonus(self, logits: jnp.ndarray) -> jnp.ndarray:
        probs = nn.softmax(logits)
        entropy_per_disribution = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
        return jnp.mean(entropy_per_disribution)


    def ppo_loss(self, logits: jnp.ndarray, old_logits: jnp.ndarray, advantages: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        return self.policy_loss(logits, old_logits, advantages, actions) - self._entropy_coefficient * self.entropy_bonus(logits)