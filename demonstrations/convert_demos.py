import gymnasium as gym
import gym_pusht
import numpy as np
import collections
import pathlib
import zarr
import cv2
import copy

# --- 1. The Wrapper (As provided) ---
class PushT(gym.Env):
    def __init__(self, size=(64,64), obs_type="pixels_agent_pos", render_mode="rgb_array", force_sparse=False, max_steps=1000, action_repeat=1, env_kwargs={}):
        w,h = size
        self._env = gym.make("gym_pusht/PushT-v0", obs_type=obs_type, render_mode=render_mode, observation_width=w, observation_height=h, force_sparse=force_sparse, display_cross=False, **env_kwargs)
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
        self._obs_key = "image"
        self.force_sparse = force_sparse
        self.max_steps = max_steps; self.nstep = 0
        self.action_repeat = action_repeat

    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        try: return getattr(self._env, name)
        except AttributeError: raise ValueError(name)
        
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
            reward = 2 * self.max_steps
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

def replay_demo_with_cache(env, episode_data, cache, ep_id):
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
    # We need to include 'action' for this step.
    t["action"] = convert(actions[0]) # Action to be taken
    add_to_cache(cache, ep_id, t)

    # --- EXECUTE ACTIONS ---
    # We have N actions. We will have N+1 observations total (0 to N).
    
    for i, action in enumerate(actions):
        # Step env
        next_obs, reward, done, info = env.step(action)
        
        transition = {k: convert(v) for k, v in next_obs.items()}
        
        # Action for this new step?
        # If this is the last step (done), action might differ or be zero.
        # If there are N actions, we produce N transitions.
        # The last observation (terminal) needs an action field too for consistency.
        
        if i < len(actions) - 1:
            transition["action"] = convert(actions[i+1])
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
    ZARR_PATH = pathlib.Path(__file__).parent / "pusht/pusht_cchi_v7_replay.zarr"
    SAVE_DIR = pathlib.Path(__file__).parent / "training_dataset_v3"
    
    print(f"Loading demos from {ZARR_PATH}")
    demos = load_pusht_demos(ZARR_PATH)
    
    # NOTE: User provided wrapper default size (64,64) but Zarr usually 96x96.
    # Pusht-v0 defaults to 96x96.
    # User's provided wrapper code in prompt had size=(64,64) in __init__.
    # But later prompt code: env = PushT(max_steps=300) without size -> defaults to 64x64.
    # Demos are usually at 96x96. If we render at 64x64, it's fine.
    
    # Let's use 96x96 to match typical PushT defaults unless strictly requested 64.
    # User's prompt code: class PushT(gym.Env): def __init__(self, size=(64,64)...
    # I changed it to 96x96 in my pasted code above to be safe, or should I stick to 64?
    # Dreamer usually likes 64x64.
    # I'll stick to 96x96 for higher quality, or make it configurable. 
    # Let's check what the user wants. "same data as the following gym-pusht wrapper".
    # The wrapper has default (64,64).
    # I will modify the script to use (96,96) because typically we want high res, 
    # but resize if needed. Actually, let's just use 96x96. 
    # Wait, I set default 96x96 in the script above.
    
    env = PushT(size=(64,64), max_steps=500, render_mode="rgb_array", env_kwargs={"pixels_based_success": True}) 

    cache = collections.OrderedDict()

    print(f"Starting collection into {SAVE_DIR}...")

    # Process all demos
    for i in range(len(demos)):
        ep_id = f"episode_{i:06d}"
        # print(f"Processing {ep_id}...")
        
        try:
            cov, pix_cov = replay_demo_with_cache(
                env, 
                copy.deepcopy(demos[i]), 
                cache, 
                ep_id
            )
            # print(f"  Cov: {cov:.2f}, PixCov: {pix_cov:.2f}")
            save_episodes(SAVE_DIR, {ep_id: cache[ep_id]})
            del cache[ep_id]
        except Exception as e:
            print(f"Failed on {ep_id}: {e}")
            # import traceback
            # traceback.print_exc()

    print("Done.")
