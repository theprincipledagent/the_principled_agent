import chex
import enum
import flax.linen as nn
import functools
import gymnax
import jax
import jax.numpy as jnp
import optax
import typing

import ppo
import visualizations


class AdvantageType(enum.Enum):
    """Defines the type of advantage modification to use."""
    BASELINE = enum.auto()
    RANDOM_NOISE = enum.auto()
    BIAS = enum.auto()


_PARALLEL_ENVS = 16
_STEPS_PER_UPDATE = 256
_TOTAL_STEPS = 1_000_000
_ACTOR_LEARNING_RATE = 0.001
_CRITIC_LEARNING_RATE = 0.001
_NUM_EVAL_EPISODES = 1024
_FINAL_NUM_EVAL_EPISODES = 1024

# PPO Hyperparameters
_CLIP_EPSILON = 0.1
_ENTROPY_COEFFICIENT = 0.05
_GAMMA = 0.95
_LAMBDA = 0.95

# Advantage type can be BASELINE, RANDOM_NOISE or BIAS
_ADVANTAGE_TYPE = AdvantageType.BASELINE
_RANDOM_NOISE_STDDEV = 0.002
_CORRELATED_NOISE_COEFFICIENT = 0.08


class Policy(nn.Module):
    action_space: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs
        x = nn.Conv(features=16, kernel_size=(3, 3,), strides=(1, 1,), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3,), strides=(1, 1,), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3,), strides=(1, 1,), padding='VALID')(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_space)(x)
        return x
    

class Policy2(nn.Module):
    action_space: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs
        x = nn.Conv(features=8, kernel_size=(3, 3,), strides=(1, 1,), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.Conv(features=8, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(4, 4,), strides=(1, 1,), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4,), strides=(1, 1,), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_space)(x)
        return x


class Value(nn.Module):
    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs
        x = nn.Conv(features=16, kernel_size=(3, 3,), strides=(1, 1,), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3,), strides=(1, 1,), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3,), strides=(1, 1,), padding='VALID')(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class Value2(nn.Module):
    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs
        x = nn.Conv(features=8, kernel_size=(3, 3,), strides=(1, 1,), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.Conv(features=8, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(4, 4,), strides=(1, 1,), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4,), strides=(1, 1,), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4,), strides=(1, 1,), padding='SAME')(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class RunnerState(typing.NamedTuple):
    actor_params: typing.Any
    critic_params: typing.Any
    actor_opt_state: typing.Any
    critic_opt_state: typing.Any
    obs: jnp.ndarray
    env_state: typing.Any
    key: typing.Any


ppo_impl = ppo.PPO(_CLIP_EPSILON, _ENTROPY_COEFFICIENT, _GAMMA, _LAMBDA, _PARALLEL_ENVS)

@functools.partial(jax.jit, static_argnames=('env', 'actor', 'num_episodes'))
def evaluate_agent(key, env, env_params, actor, actor_params, num_episodes: int):
    """Runs the agent for a number of episodes and returns the total rewards."""
    
    def play_one_episode(rng):
        """Plays one episode and returns the cumulative reward."""
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset(reset_rng, env_params)
        
        def step_in_env(carry):
            """A single step in the environment loop."""
            rng, current_obs, current_env_state, cumulative_reward, done = carry
            
            def true_fn(c):
                return c
            
            def false_fn(c):
                rng, current_obs, current_env_state, cumulative_reward, _ = c
                rng, action_rng, step_rng = jax.random.split(rng, 3)
                
                logits = actor.apply({'params': actor_params}, jnp.expand_dims(current_obs, 0))
                action = jnp.argmax(logits, axis=-1)[0]
                
                n_obs, n_env_state, reward, n_done, _ = env.step(step_rng, current_env_state, action, env_params)
                
                return rng, n_obs, n_env_state, cumulative_reward + reward, n_done

            return jax.lax.cond(done, true_fn, false_fn, carry)

        initial_carry = (rng, obs, env_state, 0.0, False)
        _, _, _, final_reward, _ = jax.lax.fori_loop(0, 1000, lambda _, val: step_in_env(val), initial_carry)
        
        return final_reward

    keys = jax.random.split(key, num_episodes)
    all_rewards = jax.vmap(play_one_episode)(keys)
    
    return jnp.mean(all_rewards), jnp.max(all_rewards), jnp.min(all_rewards)


def main():
    key = jax.random.key(7)
    keys = jax.random.split(key, _PARALLEL_ENVS)

    env, env_params = gymnax.make('Breakout-MinAtar')
    
    reset = jax.vmap(env.reset, in_axes=(0, None,))

    def auto_reset_step(rng: chex.PRNGKey,
        state: gymnax.EnvState,
        action: typing.Union[int, float],
        params: gymnax.EnvParams):

        step_key, reset_key = jax.random.split(rng)

        n_obs, n_state, reward, done, info = env.step(step_key, state, action, params)

        def reset_if_done(op):
            key, env_params = op
            return env.reset(key, env_params)

        def carry_over_if_not_done(op):
            return n_obs, n_state

        final_obs, final_state = jax.lax.cond(
            done,
            reset_if_done,
            carry_over_if_not_done,
            operand=(reset_key, params)
        )

        return final_obs, final_state, reward, done, info

    step = jax.vmap(auto_reset_step, in_axes=(0, 0, 0, None,))

    obs, state = reset(keys, env_params)

    key, actor_key, critic_key = jax.random.split(key, 3)

    actor = Policy(env.action_space().n)
    actor_params = actor.init(actor_key, obs)['params']
    actor_solver = optax.adam(learning_rate=_ACTOR_LEARNING_RATE)
    actor_opt_state = actor_solver.init(actor_params)

    critic = Value()
    critic_params = critic.init(critic_key, obs)['params']
    critic_solver = optax.adam(learning_rate=_CRITIC_LEARNING_RATE)
    critic_opt_state = critic_solver.init(critic_params)

    print(actor.tabulate(key, jnp.zeros((1, 10, 10, 4)), depth=1))

    @jax.jit
    def train_step(runner_state: RunnerState):
        def rollout(carry, _):
            obs, env_state, key = carry

            key, action_key, step_key = jax.random.split(key, 3)
            keys = jax.random.split(step_key, _PARALLEL_ENVS)

            values = critic.apply({'params': runner_state.critic_params}, obs).squeeze()
            logits = actor.apply({'params': runner_state.actor_params}, obs)
            actions = jax.random.categorical(action_key, logits)
            n_obs, n_state, reward, done, _ = step(keys, env_state, actions, env_params)

            log_probs = jnp.take_along_axis(nn.log_softmax(logits), actions[..., None], axis=-1).squeeze(-1)

            transition = ppo.Trajectory(obs, values, actions, log_probs, reward, done)
            
            return (n_obs, n_state, key), transition

        initial_carry = (runner_state.obs, runner_state.env_state, runner_state.key)
        (final_obs, final_eval_state, final_key), trajectory = jax.lax.scan(rollout, initial_carry, None, length=_STEPS_PER_UPDATE)
        
        advantages = ppo_impl.calc_advantages(trajectory.val, trajectory.reward, trajectory.done, critic.apply({'params': runner_state.critic_params}, final_obs).squeeze())
        returns = ppo_impl.calc_returns(trajectory.reward, advantages)

        advantages_wrong =  ppo_impl.calc_advantages(trajectory.val, trajectory.reward, trajectory.done, critic.apply({'params': runner_state.critic_params}, obs).squeeze())
        advantages_wrong = (advantages_wrong - advantages_wrong.mean()) / (advantages_wrong.std() + 1e-8)

        def _baseline_advantage(op):
            advantages, key, _, _, _ = op
            norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return norm_advantages, key
        
        def _random_noise_advantage(op):
            advantages, key, _, _, _ = op
            norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            noise_key, new_key = jax.random.split(key)
            noise = jax.random.normal(noise_key, norm_advantages.shape) * _RANDOM_NOISE_STDDEV
            return norm_advantages + noise, new_key
        
        def _bias_advantage(op):
            advantages, key, critic_params, obs, critic_model = op
            
            bias_advantages = advantages + _CORRELATED_NOISE_COEFFICIENT * critic_model.apply({'params': critic_params}, obs).squeeze()
            norm_advantages = (bias_advantages - bias_advantages.mean()) / (bias_advantages.std() + 1e-8)
            return norm_advantages, key
        
        operand = (advantages, final_key, runner_state.critic_params, obs, critic)
    
        advantages, final_key = jax.lax.switch(
            index=_ADVANTAGE_TYPE.value - 1,
            branches=[
                _baseline_advantage,
                _random_noise_advantage,
                _bias_advantage
            ],
            operand=operand
        )

        def actor_loss_fn(actor_params):
            new_logits = actor.apply({'params': actor_params}, trajectory.obs.reshape(-1, 10, 10, 4))
            return ppo_impl.ppo_loss(new_logits, trajectory.log_prob.flatten(), advantages.flatten(), trajectory.action.flatten())

        actor_loss, actor_grad = jax.value_and_grad(actor_loss_fn)(runner_state.actor_params)
        actor_updates, actor_opt_state = actor_solver.update(actor_grad, runner_state.actor_opt_state, runner_state.actor_params)
        actor_params = optax.apply_updates(runner_state.actor_params, actor_updates)

        def critic_loss_fn(critic_params):
            values = critic.apply({'params': critic_params}, trajectory.obs.reshape(-1, 10, 10, 4))
            return ppo_impl.value_loss(values, returns.flatten())

        critic_loss, critic_grad = jax.value_and_grad(critic_loss_fn)(runner_state.critic_params)
        critic_updates, critic_opt_state = critic_solver.update(critic_grad, runner_state.critic_opt_state, runner_state.critic_params)
        critic_params = optax.apply_updates(runner_state.critic_params, critic_updates)

        new_runner_state = RunnerState(
            actor_params=actor_params,
            critic_params=critic_params,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            obs=final_obs,
            env_state=final_eval_state,
            key=final_key
        )

        metrics = {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'avg_reward': trajectory.reward.mean(),
            'avg_entropy': ppo_impl.entropy_bonus(actor.apply({'params': actor_params}, trajectory.obs.reshape(-1, 10, 10, 4))),
            'actor_grad': actor_grad,
            'advantages_min': advantages.min(),
            'advantages_max': advantages.max(),
            'advantages_wrong_min': advantages_wrong.min(),
            'advantages_wrong_max': advantages_wrong.max(),
        }

        return new_runner_state, metrics
    
    runner_state = RunnerState(actor_params, critic_params, actor_opt_state, critic_opt_state, obs, state, key)

    actor_losses = []
    critic_losses = []
    avg_step_rewards = []
    avg_episode_rewards = []
    avg_episode_reward_steps = []
    entropies = []
    advantages_min = []
    advantages_max = []
    advantages_wrong_min = []
    advantages_wrong_max = []
    grad_norms = {}

    for i in range(0, _TOTAL_STEPS, _STEPS_PER_UPDATE):
        runner_state, metrics = train_step(runner_state)

        actor_losses.append(metrics['actor_loss'])
        critic_losses.append(metrics['critic_loss'])
        avg_step_rewards.append(metrics['avg_reward'])
        entropies.append(metrics['avg_entropy'])
        advantages_min.append(metrics['advantages_min'])
        advantages_max.append(metrics['advantages_max'])
        advantages_wrong_min.append(metrics['advantages_wrong_min'])
        advantages_wrong_max.append(metrics['advantages_wrong_max'])

        actor_grads = metrics['actor_grad']
        for layer_name in actor_grads.keys():
            if layer_name not in grad_norms:
                grad_norms[layer_name] = []
            
            layer_grad_tree = actor_grads[layer_name]
            flat_grads = jnp.concatenate([g.flatten() for g in jax.tree_util.tree_leaves(layer_grad_tree)])
            norm = jnp.linalg.norm(flat_grads)
            grad_norms[layer_name].append(norm)
                
        
        if i % 300 * _STEPS_PER_UPDATE == 0:
            key, eval_key = jax.random.split(runner_state.key)
            runner_state = runner_state._replace(key=key)
            
            avg_episode_score, _, _ = evaluate_agent(
                eval_key, env, env_params, actor, runner_state.actor_params, _NUM_EVAL_EPISODES
            )
            avg_episode_rewards.append(avg_episode_score)

            avg_episode_reward_steps.append(len(actor_losses))

            print(f"Update {i}/{_TOTAL_STEPS}:")
            print(f"  Actor Loss: {metrics['actor_loss']:.4f}")
            print(f"  Critic Loss: {metrics['critic_loss']:.4f}")
            print(f"  Average Step Reward: {metrics['avg_reward']:.4f}")
            print(f"  Average Episode Reward: {avg_episode_score:.2f}")
            print(f"  Avg Entropy: {metrics['avg_entropy']:.4f}")

    key, eval_key = jax.random.split(runner_state.key)
    avg_episode_score, max_episode_score, min_episode_score = evaluate_agent(
        eval_key, env, env_params, actor, runner_state.actor_params, _FINAL_NUM_EVAL_EPISODES
    )

    print(f"Final Average Episode Reward: {avg_episode_score:.2f}")
    print(f"Final Max Episode Reward: {max_episode_score:.2f}")
    print(f"Final Min Episode Reward: {min_episode_score:.2f}")

    visualizations.plot_advantages(advantages_min, advantages_max)
    visualizations.plot_advantages(advantages_wrong_min, advantages_wrong_max, filename='advantages_wrong.png')
    visualizations.plot_advantages(jnp.subtract(jnp.array(advantages_min), jnp.array(advantages_wrong_min)).tolist(), jnp.subtract(jnp.array(advantages_max), jnp.array(advantages_wrong_max)).tolist(), filename='advantages_diff.png')
    visualizations.policy_value_and_reward_chart(actor_losses, critic_losses, avg_episode_rewards, avg_episode_reward_steps)
    visualizations.generate_full_agent_playback(key, env, env_params, actor, runner_state.actor_params, critic, runner_state.critic_params)
    visualizations.generate_full_agent_playback(key, env, env_params, actor, runner_state.actor_params, critic, runner_state.critic_params, with_salience=True, filename='agent_playback_salience.mp4')
    visualizations.loss_location_heatmap(key, env, env_params, actor, runner_state.actor_params)
    visualizations.entropy_chart(entropies, env.action_space().n)
    visualizations.entropy_location_heatmap(key, env, env_params, actor, runner_state.actor_params)
    visualizations.cnn_filter_visualization(runner_state.actor_params)
    visualizations.plot_gradient_norms(grad_norms)


if __name__ == '__main__':
    main()