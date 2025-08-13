import cv2 as cv
import flax.linen as nn
import functools
import jax
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

_DPI = 300

_color_palette = {
    "primary": "#0A58CA",   # Strong Blue
    "secondary": "#6A7C92", # Slate Gray
    "accent": "#6F42C1",    # Digital Violet
    "background": "#F8F9FA",# Alabaster White
    "text": "#212529",      # Dark Charcoal
}

_custom_cmap_mpl = mcolors.LinearSegmentedColormap.from_list(
    'maximizingreward_cmap_mpl', 
    [_color_palette['background'], _color_palette['accent']]
)


def hex_to_rgb(hex_code):
        hex_code = hex_code.lstrip('#')
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


def policy_value_and_reward_chart(
        actor_losses: list[float],
        critic_losses: list[float],
        avg_rewards: list[float],
        filename: str = 'losses_and_rewards.png'):
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    fig.set_facecolor(_color_palette["background"])

    ax1.plot(actor_losses, color=_color_palette["accent"], linewidth=2)
    ax1.set_title('Actor Loss Over Time', color=_color_palette["text"], fontsize=16, weight='bold')
    ax1.set_ylabel('Loss', color=_color_palette["text"], fontsize=12)
    ax1.set_facecolor(_color_palette["background"])
    ax1.tick_params(colors=_color_palette["text"])
    ax1.spines['bottom'].set_color(_color_palette["secondary"])
    ax1.spines['left'].set_color(_color_palette["secondary"])

    ax2.plot(critic_losses, color=_color_palette["accent"], linewidth=2)
    ax2.set_title('Critic Loss Over Time', color=_color_palette["text"], fontsize=16, weight='bold')
    ax2.set_ylabel('Loss', color=_color_palette["text"], fontsize=12)
    ax2.set_facecolor(_color_palette["background"])
    ax2.tick_params(colors=_color_palette["text"])
    ax2.spines['bottom'].set_color(_color_palette["secondary"])
    ax2.spines['left'].set_color(_color_palette["secondary"])

    ax3.plot(avg_rewards, color=_color_palette["accent"], linewidth=2)
    ax3.set_title('Average Reward Over Time', color=_color_palette["text"], fontsize=16, weight='bold')
    ax3.set_ylabel('Reward', color=_color_palette["text"], fontsize=12)
    ax3.set_xlabel('Training Update Step', color=_color_palette["text"], fontsize=12)
    ax3.set_facecolor(_color_palette["background"])
    ax3.tick_params(colors=_color_palette["text"])
    ax3.spines['bottom'].set_color(_color_palette["secondary"])
    ax3.spines['left'].set_color(_color_palette["secondary"])

    plt.tight_layout(pad=3.0)
    plt.savefig(
        filename,
        dpi=_DPI,
        bbox_inches='tight',
        facecolor=fig.get_facecolor()
    )


def generate_single_playback_frame(obs, actor, actor_params, critic, critic_params, value_history: list[float], entropy_history: list[float], with_salience: bool = False):
    values = critic.apply({'params': critic_params}, jnp.expand_dims(obs, 0)).squeeze()
    logits = actor.apply({'params': actor_params}, jnp.expand_dims(obs, 0)).squeeze()

    max_entropy = math.log2(logits.shape[-1])

    saliency_map = None
    if with_salience:
        def get_logit_for_action(actor_params, obs):
            all_logits = actor.apply({'params': actor_params}, jnp.expand_dims(obs, 0)).squeeze()
            action = jnp.argmax(all_logits)
            return all_logits[action]
        
        grad_fn = jax.grad(get_logit_for_action, argnums=1)

        saliency_map = grad_fn(actor_params, obs)
        saliency_map = jnp.sum(jnp.abs(saliency_map), axis=-1)
        if jnp.max(saliency_map) > 0:
            saliency_map = saliency_map / jnp.max(saliency_map)

    value_history.append(values)

    plt.style.use('seaborn-v0_8-whitegrid')
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(2, 3), constrained_layout=True)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(2, 4.5), constrained_layout=True)
    fig.set_facecolor(_color_palette["background"])

    ax1.bar(np.arange(0, len(logits)).astype('str'), jax.nn.softmax(logits), color=_color_palette['accent'])
    ax1.set_facecolor(_color_palette["background"])
    ax1.tick_params(colors=_color_palette["text"])
    ax1.spines['bottom'].set_color(_color_palette["secondary"])
    ax1.spines['left'].set_color(_color_palette["secondary"])
    ax1.set_ylim(0, 1)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_title('Actions')

    ax2.plot(value_history, color=_color_palette["accent"], linewidth=2)
    ax2.set_facecolor(_color_palette["background"])
    ax2.tick_params(colors=_color_palette["text"])
    ax2.spines['bottom'].set_color(_color_palette["secondary"])
    ax2.spines['left'].set_color(_color_palette["secondary"])
    ax2.set_ylim(0, 1)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_title('Value Estimation')

    ax3.plot(entropy_history, color=_color_palette["accent"], linewidth=2)
    ax3.set_facecolor(_color_palette["background"])
    ax3.tick_params(colors=_color_palette["text"])
    ax3.spines['bottom'].set_color(_color_palette["secondary"])
    ax3.spines['left'].set_color(_color_palette["secondary"])
    ax3.set_ylim(0, max_entropy * 1.2)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_title('Uncertainty')
    ax3.axhline(
        y=max_entropy,
        color=_color_palette["secondary"],
        linestyle='--',
        linewidth=2,
        label='Max Entropy'
    )

    fig.canvas.draw()
    action_prob_buffer = fig.canvas.buffer_rgba()
    cols, rows = fig.canvas.get_width_height()
    action_prob_img_array = np.frombuffer(action_prob_buffer, dtype=np.uint8).reshape(rows, cols, 4)
    action_prob_img_array = cv.cvtColor(action_prob_img_array, cv.COLOR_RGBA2BGR)
    plt.close(fig)

    channel_colors = [
        hex_to_rgb(_color_palette["primary"]),   # Channel 0: Paddle (Blue)
        hex_to_rgb(_color_palette["accent"]),    # Channel 1: Ball (Violet)
        hex_to_rgb(_color_palette["accent"]),    # Channel 2: Ball (Violet)
        hex_to_rgb(_color_palette["secondary"]), # Channel 3: Bricks (Gray)
    ]

    scale = 50
    height, width, _ = obs.shape
    
    background_rgb = hex_to_rgb(_color_palette["background"])
    rgb_array = np.full((height, width, 3), background_rgb, dtype=np.uint8)

    for channel_index in [3, 0, 1, 2]:
        mask = obs[:, :, channel_index] == 1
        rgb_array[mask] = channel_colors[channel_index]

    if saliency_map is not None:
        saliency_color_rgb = np.array(hex_to_rgb(_color_palette["primary"]))
        saliency_alpha = np.array(saliency_map)[..., np.newaxis]
        rgb_array = (saliency_color_rgb * saliency_alpha + rgb_array * (1 - saliency_alpha)).astype(np.uint8)

    scaled_image = np.kron(rgb_array, np.ones((scale, scale, 1))).astype(np.uint8)
    
    frame = cv.cvtColor(scaled_image, cv.COLOR_RGB2BGR)

    target_height = 500
    game_h, game_w, _ = frame.shape
    plot_h, plot_w, _ = action_prob_img_array.shape
    new_game_w = int(game_w * (target_height / game_h))
    new_plot_w = int(plot_w * (target_height / plot_h))
    frame = cv.resize(frame, (new_game_w, target_height))
    action_prob_img_array = cv.resize(action_prob_img_array, (new_plot_w, target_height))
    total_width = new_game_w + new_plot_w
    canvas = np.zeros((target_height, total_width, 3), dtype=np.uint8)
    canvas[0:target_height, 0:new_game_w] = frame
    canvas[0:target_height, new_game_w:total_width] = action_prob_img_array

    return canvas, value_history

def generate_full_agent_playback(key, env, env_params, actor, actor_params, critic, critic_params, filename: str = 'agent_playback.mp4', with_salience: bool = False):
    key, reset_key = jax.random.split(key, 2)
    obs, env_state = env.reset(reset_key, env_params)

    value_history = []
    frames = []
    entropy_history = []

    while True:
        frame, value_history = generate_single_playback_frame(obs, actor, actor_params, critic, critic_params, value_history, entropy_history, with_salience)
        frames.append(frame)

        key, action_key, step_key = jax.random.split(key, 3)

        logits = actor.apply({'params': actor_params}, jnp.expand_dims(obs, 0))
        actions = jnp.argmax(logits, axis=-1)[0]
        obs, env_state, _, done, _ = env.step(step_key, env_state, actions, env_params)

        probs = jax.nn.softmax(logits)
        entropy = -jnp.sum(probs * jnp.log2(probs + 1e-10))

        entropy_history.append(entropy)

        if done:
            break

    height, width, _ = frames[0].shape
    frame_size = (width, height)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_writer = cv.VideoWriter(filename, fourcc, 4, frame_size)

    for frame in frames:
        if frame.shape[2] == 4: # RGBA
            frame = cv.cvtColor(frame, cv.COLOR_RGBA2BGR)
        video_writer.write(frame)

    video_writer.release()


@functools.partial(jax.jit, static_argnames=('env', 'actor', 'iterations'))
def _compute_loss_map(key, env, env_params, actor, actor_params, iterations: int = 100):
    keys = jax.random.split(key, iterations)
    
    def one_full_run(key):
        key, reset_key = jax.random.split(key, 2)
        obs, env_state = env.reset(reset_key, env_params)

        def rollout(carry, _):
            _, _, _, prev_done = carry

            def step_fn(c):
                run_key, prev_obs, prev_env_state, prev_done = c

                key, action_key, step_key = jax.random.split(run_key, 3)

                logits = actor.apply({'params': actor_params}, jnp.expand_dims(prev_obs, 0))
                actions = jax.random.categorical(action_key, logits)[0]
                n_obs, env_state, _, done, _ = env.step(step_key, prev_env_state, actions, env_params)

                obs_for_carry = jax.lax.select(done, prev_obs, n_obs)

                return (key, obs_for_carry, env_state, done), obs_for_carry
            
            def no_op_fn(c):
                run_key, prev_obs, prev_env_state, prev_done = c
                return c, prev_obs
            
            new_carry, final_obs = jax.lax.cond(
                prev_done,
                no_op_fn,
                step_fn,
                carry
            )
            return new_carry, final_obs

        initial_carry = (key, obs, env_state, False)

        _, obs = jax.lax.scan(rollout, initial_carry, None, length=1_000)

        return obs[-1]


    runs = jax.vmap(one_full_run)

    final_obs_batch = runs(keys)
    ball_loss_map = jnp.sum(final_obs_batch[:, :, :, 1], axis=0) + jnp.sum(final_obs_batch[:, :, :, 2], axis=0)
    paddle_loss_map = jnp.sum(final_obs_batch[:, :, :, 0], axis=0)

    return ball_loss_map, paddle_loss_map


def loss_location_heatmap(key, env, env_params,  actor, actor_params, iterations: int = 100, filename: str = 'loss_heatmap.png'):
    ball_loss_map, paddle_loss_map = _compute_loss_map(key, env, env_params, actor, actor_params, iterations)

    plt.style.use('seaborn-v0_8-white')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    fig.set_facecolor(_color_palette["background"])

    ax1.imshow(ball_loss_map, cmap=_custom_cmap_mpl)
    ax1.set_title('Ball Locations Heatmap', color=_color_palette["text"], fontsize=16, weight='bold')
    ax1.set_facecolor(_color_palette["background"])
    ax1.tick_params(colors=_color_palette["text"])
    ax1.spines['top'].set_color(_color_palette["secondary"])
    ax1.spines['bottom'].set_color(_color_palette["secondary"])
    ax1.spines['left'].set_color(_color_palette["secondary"])
    ax1.spines['right'].set_color(_color_palette["secondary"])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    ax2.imshow(paddle_loss_map, cmap=_custom_cmap_mpl)
    ax2.set_title('Paddle Location Heatmap', color=_color_palette["text"], fontsize=16, weight='bold')
    ax2.set_facecolor(_color_palette["background"])
    ax2.tick_params(colors=_color_palette["text"])
    ax2.spines['top'].set_color(_color_palette["secondary"])
    ax2.spines['bottom'].set_color(_color_palette["secondary"])
    ax2.spines['left'].set_color(_color_palette["secondary"])
    ax2.spines['right'].set_color(_color_palette["secondary"])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    plt.savefig(
        filename,
        dpi=_DPI,
        bbox_inches='tight',
        facecolor=fig.get_facecolor()
    )


def entropy_chart(entropy: list[float], total_actions: int, filename: str = 'entropy.png'):
    plt.style.use('seaborn-v0_8-whitegrid')

    max_entropy = math.log2(total_actions)

    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 6))

    fig.set_facecolor(_color_palette["background"])

    ax1.plot(entropy, color=_color_palette["accent"], linewidth=2)
    ax1.set_title('Action Entropy Over Time', color=_color_palette["text"], fontsize=16, weight='bold')
    ax1.set_ylabel('Loss', color=_color_palette["text"], fontsize=12)
    ax1.set_facecolor(_color_palette["background"])
    ax1.tick_params(colors=_color_palette["text"])
    ax1.spines['bottom'].set_color(_color_palette["secondary"])
    ax1.spines['left'].set_color(_color_palette["secondary"])

    ax1.axhline(
        y=max_entropy,
        color=_color_palette["secondary"],
        linestyle='--',
        linewidth=2,
        label='Max Entropy'
    )

    plt.tight_layout(pad=3.0)
    plt.savefig(
        filename,
        dpi=_DPI,
        bbox_inches='tight',
        facecolor=fig.get_facecolor()
    )


@functools.partial(jax.jit, static_argnames=('env', 'actor', 'iterations'))
def _compute_average_entropy_map(key, env, env_params, actor, actor_params, iterations: int = 100):
    keys = jax.random.split(key, iterations)
    
    def one_full_run(key):
        key, reset_key = jax.random.split(key, 2)
        obs, env_state = env.reset(reset_key, env_params)

        def rollout(carry, _):
            run_key, prev_obs, prev_env_state, prev_done, ball_e_map, paddle_e_map, ball_v_map, paddle_v_map = carry

            def step_fn(c):
                run_key, prev_obs, prev_env_state, _, prev_ball_e, prev_paddle_e, prev_ball_v, prev_paddle_v = c

                key, action_key, step_key = jax.random.split(run_key, 3)

                logits = actor.apply({'params': actor_params}, jnp.expand_dims(prev_obs, 0)).squeeze(0)
                actions = jax.random.categorical(action_key, logits)
                n_obs, n_env_state, _, done, _ = env.step(step_key, prev_env_state, actions, env_params)
                
                probs = jax.nn.softmax(logits)
                entropy = -jnp.sum(probs * jnp.log2(probs + 1e-10))

                ball_mask = prev_obs[:, :, 1] + prev_obs[:, :, 2]
                paddle_mask = prev_obs[:, :, 0]
                
                ball_e_map = prev_ball_e + entropy * ball_mask
                paddle_e_map = prev_paddle_e + entropy * paddle_mask

                ball_v_map = prev_ball_v + ball_mask
                paddle_v_map = prev_paddle_v + paddle_mask

                return (key, n_obs, n_env_state, done, ball_e_map, paddle_e_map, ball_v_map, paddle_v_map), None
            
            def no_op_fn(c):
                return c, None
            
            new_carry, _ = jax.lax.cond(
                prev_done,
                no_op_fn,
                step_fn,
                carry
            )
            return new_carry, None
        
        ball_entropy_map = jnp.zeros_like(obs[:, :, 0])
        paddle_entropy_map = jnp.zeros_like(obs[:, :, 0])
        ball_visit_map = jnp.zeros_like(obs[:, :, 0])
        paddle_visit_map = jnp.zeros_like(obs[:, :, 0])

        initial_carry = (key, obs, env_state, False, ball_entropy_map, paddle_entropy_map, ball_visit_map, paddle_visit_map)

        (_, _, _, _, final_ball_e, final_paddle_e, final_ball_v, final_paddle_v), _ = jax.lax.scan(rollout, initial_carry, None, length=1_000)

        return final_ball_e, final_paddle_e, final_ball_v, final_paddle_v

    ball_e_maps, paddle_e_maps, ball_v_maps, paddle_v_maps = jax.vmap(one_full_run)(keys)

    total_ball_entropy = jnp.sum(ball_e_maps, axis=0)
    total_paddle_entropy = jnp.sum(paddle_e_maps, axis=0)
    total_ball_visits = jnp.sum(ball_v_maps, axis=0)
    total_paddle_visits = jnp.sum(paddle_v_maps, axis=0)
    
    avg_ball_entropy = total_ball_entropy / (total_ball_visits + 1e-10)
    avg_paddle_entropy = total_paddle_entropy / (total_paddle_visits + 1e-10)

    return avg_ball_entropy, avg_paddle_entropy


def entropy_location_heatmap(key, env, env_params,  actor, actor_params, iterations: int = 100, filename: str = 'entropy_heatmap.png'):
    ball_entropy_map, paddle_entropy_map = _compute_average_entropy_map(key, env, env_params, actor, actor_params, iterations)

    plt.style.use('seaborn-v0_8-white')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    fig.set_facecolor(_color_palette["background"])

    ax1.imshow(ball_entropy_map, cmap=_custom_cmap_mpl)
    ax1.set_title('Ball Locations Heatmap', color=_color_palette["text"], fontsize=16, weight='bold')
    ax1.set_facecolor(_color_palette["background"])
    ax1.tick_params(colors=_color_palette["text"])
    ax1.spines['top'].set_color(_color_palette["secondary"])
    ax1.spines['bottom'].set_color(_color_palette["secondary"])
    ax1.spines['left'].set_color(_color_palette["secondary"])
    ax1.spines['right'].set_color(_color_palette["secondary"])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    ax2.imshow(paddle_entropy_map, cmap=_custom_cmap_mpl)
    ax2.set_title('Paddle Location Heatmap', color=_color_palette["text"], fontsize=16, weight='bold')
    ax2.set_facecolor(_color_palette["background"])
    ax2.tick_params(colors=_color_palette["text"])
    ax2.spines['top'].set_color(_color_palette["secondary"])
    ax2.spines['bottom'].set_color(_color_palette["secondary"])
    ax2.spines['left'].set_color(_color_palette["secondary"])
    ax2.spines['right'].set_color(_color_palette["secondary"])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    plt.savefig(
        filename,
        dpi=_DPI,
        bbox_inches='tight',
        facecolor=fig.get_facecolor()
    )
