import gymnasium as gym
import gym_pusht
import numpy as np
import collections
import pathlib
import zarr
import cv2
import copy
import argparse

PUSHT_STATIC_SUCCESS_REWARD = 300 # corresponds to max_steps=300

# --- 1. The Wrapper (As provided) ---
class PushT(gym.Env):
    def __init__(self, size=(64,64), obs_type="pixels_agent_pos", render_mode="rgb_array", force_sparse=False, max_steps=1000, action_repeat=1, use_differential_action=False, env_kwargs={}):
        w,h = size
        self._env = gym.make("gym_pusht/PushT-v0", obs_type=obs_type, render_mode=render_mode, observation_width=w, observation_height=h, force_sparse=force_sparse, display_cross=False, differential_action=use_differential_action, **env_kwargs)
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
        self._obs_key = "image"
        self.force_sparse = force_sparse
        self.max_steps = max_steps; self.nstep = 0
        self.action_repeat = action_repeat
        self.action_space = self._env.action_space

    @property
    def differential_action(self):
        return self._env.unwrapped.differential_action

    def step(self, action):
        for _ in range(self.action_repeat):
            obs, reward, done, truncated, info = self._env.step(action)
            if done: break

        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        else:
            for k,v in obs.items():
                obs[k] = np.array(v)

        if "image" not in obs and "pixels" in obs: obs["image"] = obs.pop("pixels")
        if "agent_pos" in obs and "state" not in obs: obs["state"] = obs.pop("agent_pos")
        if "pixels" in obs: obs.pop('pixels')

        info['success'] = np.array(info.get('is_success', False))
        info['coverage'] = np.array(info.get('coverage', 0.0))

        if info["is_success"]:
            reward = PUSHT_STATIC_SUCCESS_REWARD
            # print("Success!")

        if self.force_sparse:
            reward = 100.0 if info['is_success'] else 0.0

        obs["is_first"] = False
        obs["is_last"] = done
        obs["is_terminal"] = info.get("is_terminal", False)

        self.nstep += 1
        return obs, reward, done, info

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset()
        if seed is not None:
             # Try to seed deeper if necessary, but usually top level is enough
            try:
                obs, info = self._env.reset(seed=seed)
            except:
                pass
                
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}

        if "image" not in obs and "pixels" in obs: obs["image"] = obs.pop("pixels")
        if "agent_pos" in obs and "state" not in obs: obs["state"] = obs.pop("agent_pos")
        if "pixels" in obs: obs.pop('pixels')

        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        self.nstep = 0
        return obs

# --- 2. Caching Helpers ---

def convert(value, precision=32):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        return value
    return value.astype(dtype)

def add_to_cache(cache, id, transition):
    if id not in cache:
        cache[id] = dict()
        for key, val in transition.items():
            cache[id][key] = [convert(val)]
    else:
        for key, val in transition.items():
            if key not in cache[id]:
                # Initialize with zeros if missing in previous steps (shouldn't happen often)
                cache[id][key] = [convert(0 * val)] * (len(list(cache[id].values())[0]) - 1)
                cache[id][key].append(convert(val))
            else:
                cache[id][key].append(convert(val))

def save_episodes(directory, episodes):
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for ep_id, episode_data in episodes.items():
        np_data = {k: np.array(v) for k, v in episode_data.items()}
        num_steps = len(np_data['reward'])
        print(f"Saving {ep_id} to {directory} ({num_steps} steps)")
        np.savez_compressed(directory / f"{ep_id}-{num_steps}.npz", **np_data)

# --- 3. Replay Logic ---

def normalize_action(action, action_space):
    """Normalize action to [-1, 1] based on action_space bounds."""
    low = action_space.low
    high = action_space.high
    return 2.0 * (action - low) / (high - low) - 1.0

def replay_demo_with_cache(env, episode_data, cache, ep_id, normalize=False):
    initial_state = episode_data['state'][0]
    
    # Reset Wrapper (gets is_first=True)
    obs = env.reset()
    
    # We need to access the Base Env to set the physical state
    # env._env is usually TimeLimit.
    # We want PushTEnv.
    base_env = env._env.unwrapped 
    
    # Force state
    # Calling base_env.reset helps if needed, but we want to keep wrapper's is_first state
    base_env.reset(options={"reset_to_state": initial_state})
    
    # Settle physics (as done in notebook)
    for _ in range(100):
        base_env._set_state(initial_state)

    # Re-render to get the image matching the forced state
    img = base_env.render()
    current_agent_pos = np.array(base_env.agent.position)
    
    # Update the initial observation with the correct image state
    obs["image"] = img
    obs["state"] = current_agent_pos
    
    # Add Initial State (Step 0)
    t = {k: convert(v) for k, v in obs.items()}
    t["reward"] = 0.0
    t["discount"] = 1.0
    # Dummy action for first step? No, actions are usually aligned with transitions.
    # Usually Step 0 has is_first=True, action=0 (or first action), reward=0.
    # Dreamer format:
    # Step t: obs[t], action[t], reward[t], discount[t], is_first[t], is_last[t]
    # action[t] is the action taken AFTER obs[t].
    
    # We will populate action in the loop. For now, placeholder or handle later.
    # Actually, add_to_cache appends lists. We sync them at the end or step by step.
    
    # CAUTION: The action at index i in demo corresponds to transition from state i to i+1.
    # So for step 0 (initial state), we record action[0].
    
    # Let's verify demo length
    actions = episode_data['action']
    
    # Store Step 0 (observation)
    last_target = current_agent_pos
    if env.differential_action:
        # First delta is action[0] - initial_agent_pos
        first_step_action = actions[0] - last_target
    else:
        first_step_action = actions[0]

    if normalize:
        saved_action = normalize_action(first_step_action, env.action_space)
    else:
        saved_action = first_step_action

    t["action"] = convert(saved_action) 
    add_to_cache(cache, ep_id, t)

    # --- EXECUTE ACTIONS ---
    # We have N actions. We will have N+1 observations total (0 to N).
    
    # last_target is already current_agent_pos
    for i, action in enumerate(actions):
        if env.differential_action:
            # step_action is the delta to reach the target 'action'
            step_action = action - last_target
            # last_target moves to the target just reached
            last_target = action
        else:
            step_action = action

        # Step env
        next_obs, reward, done, info = env.step(step_action)
        
        transition = {k: convert(v) for k, v in next_obs.items()}
        
        if i < len(actions) - 1:
            # Action for the NEXT step (the transition from obs[i+1] to obs[i+2])
            next_target = actions[i+1]
            if env.differential_action:
                step_next_action = next_target - last_target
            else:
                step_next_action = next_target
            
            if normalize:
                saved_next_action = normalize_action(step_next_action, env.action_space)
            else:
                saved_next_action = step_next_action

            transition["action"] = convert(saved_next_action)
        else:
            # End of demo. Pad with zero action.
            transition["action"] = convert(np.zeros_like(action))
            
        transition["reward"] = reward
        transition["discount"] = 1.0 - float(done)
        
        add_to_cache(cache, ep_id, transition)
        
        if done:
            break
            
    # Note: Demos usually end with success.
    # info['coverage'] and pixel coverage check
    pixel_coverage = base_env._get_pixel_coverage()
    return info.get('coverage', 0.0), pixel_coverage

# --- 4. Main Execution ---

def load_pusht_demos(zarr_path):
    root = zarr.open(zarr_path, mode='r')
    all_actions = root['data/action'][:]
    all_states = root['data/state'][:]
    episode_ends = root['meta/episode_ends'][:]
    episodes = []
    start_idx = 0
    for end_idx in episode_ends:
        episodes.append({
            'action': all_actions[start_idx:end_idx],
            'state': all_states[start_idx:end_idx],
        })
        start_idx = end_idx
    return episodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", type=str, default=str(pathlib.Path(__file__).parent / "pusht/pusht_cchi_v7_replay.zarr"))
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=206)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--use_differential", action="store_true", default=True)
    parser.add_argument("--normalize_actions", action="store_true", help="Normalize actions to [-1, 1] based on env action space")
    args = parser.parse_args()

    ZARR_PATH = pathlib.Path(args.zarr_path)
    print(f"Loading demos from {ZARR_PATH}")
    demos = load_pusht_demos(ZARR_PATH)
    
    use_differential_action = args.use_differential
    
    if args.save_dir:
        SAVE_DIR = pathlib.Path(args.save_dir)
    else:
        name = "training_dataset_v3"
        if use_differential_action: 
            name += "_differential"
        if args.normalize_actions:
            name += "_normalized"
        SAVE_DIR = pathlib.Path(__file__).parent / name
    
    env = PushT(size=(args.resolution, args.resolution), max_steps=500, use_differential_action=use_differential_action, render_mode="rgb_array", env_kwargs={"pixels_based_success": True}) 

    cache = collections.OrderedDict()

    print(f"Starting collection into {SAVE_DIR} (normalize={args.normalize_actions})...")

    # Process demos for evaluation
    num_to_eval = min(args.num_episodes, len(demos))
    success_count = 0
    pix_success_count = 0
    
    failed_episodes = []
    for i in range(num_to_eval):
        ep_id = f"episode_{i:06d}"
        
        try:
            cov, pix_cov = replay_demo_with_cache(
                env, 
                copy.deepcopy(demos[i]), 
                cache, 
                ep_id,
                normalize=args.normalize_actions
            )
            is_success = cov > env._env.unwrapped.success_threshold
            if is_success:
                success_count += 1
            else:
                failed_episodes.append((ep_id, cov))

            if pix_cov > env._env.unwrapped.success_threshold:
                pix_success_count += 1
                
            save_episodes(SAVE_DIR, {ep_id: cache[ep_id]})
            del cache[ep_id]
        except Exception as e:
            print(f"Failed on {ep_id}: {e}")

    print(f"Evaluation complete for {num_to_eval} episodes.")
    print(f"Success Rate (Geometric): {success_count / num_to_eval * 100:.2f}%")
    print(f"Success Rate (Pixel): {pix_success_count / num_to_eval * 100:.2f}%")
    if failed_episodes:
        print("Failed Episodes (ID, Max Coverage):")
        for fid, fcov in failed_episodes:
            print(f"  {fid}: {fcov:.3f}")
