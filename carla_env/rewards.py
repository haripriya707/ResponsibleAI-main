import numpy as np
from config import CONFIG
import carla

reward_functions = {}

def create_reward_fn(reward_fn):
    """
    Wraps a reward function to handle terminal conditions and logging.
    """
    def func(env):
        terminal_reason = "Running..."
        
        # Stop if collision with another vehicle or object
        if env.collision_detected:
            env.terminal_state = True
            terminal_reason = "Collision detected"

        # Calculate reward
        reward = 0
        if not env.terminal_state:
            reward += reward_fn(env)  # Pass only the environment
        else:
            reward += -50 # Penalty for collision/failure
            print(f"{env.episode_idx}| Terminal: ", terminal_reason)

        if env.success_state:
            print(f"{env.episode_idx}| Success")
            reward += 100

        env.extra_info.extend([
            terminal_reason,
            ""
        ])
        return reward

    return func

class EmotionState:
    """
    Tracks emotional state based on RL dynamics and environmental feedback.
    Emotions:
    - Calmness: Function of smoothness (low jerk, consistent actions).
    - Restlessness: Function of control oscillation (alternating gas/brake).
    - Urgency: Function of delay/stagnation (low speed when allowed to move).
    - Frustration: Function of negative reward trend or getting stuck.
    """
    def __init__(self):
        # State variables
        self.calmness = 0.5
        self.restlessness = 0.0
        self.urgency = 0.0
        self.frustration = 0.0
        
        # History for computation
        self.history_len = 20
        self.action_history = [] # list of (throttle, brake)
        self.reward_history = []
        self.speed_history = []
        
        # Hyperparameters (Weights)
        self.w_calm = 0.5
        self.w_restless = -0.5
        self.w_urgency = -0.2
        self.w_frustion = -0.5
        
        # Scale for total emotion reward
        self.global_scale = 1.0 

    def update(self, env, current_base_reward):
        # 1. Update History
        throttle = env.vehicle.control.throttle
        brake = env.vehicle.control.brake
        speed = env.vehicle.get_speed()
        
        self.action_history.append((throttle, brake))
        self.reward_history.append(current_base_reward)
        self.speed_history.append(speed)
        
        if len(self.action_history) > self.history_len:
            self.action_history.pop(0)
            self.reward_history.pop(0)
            self.speed_history.pop(0)
            
        # 2. Compute Emotions
        
        # --- Calmness (Smoothness) ---
        # Derivative of action (Jerk-like proxy)
        if len(self.action_history) >= 2:
            prev_t, prev_b = self.action_history[-2]
            delta_action = abs(throttle - prev_t) + abs(brake - prev_b)
            # Decay towards 1.0 if smooth, drop if jerky
            self.calmness = 0.9 * self.calmness + 0.1 * (1.0 - min(delta_action, 1.0))
        else:
            self.calmness = 0.5
            
        # --- Restlessness (Oscillation) ---
        # Check for rapid switching between throttle and brake
        if len(self.action_history) >= 5:
            switches = 0
            for i in range(1, len(self.action_history)):
                t, b = self.action_history[i]
                pt, pb = self.action_history[i-1]
                # Switch detected if dominant control changes
                if (t > 0 and pb > 0) or (b > 0 and pt > 0):
                    switches += 1
            oscillation_rate = switches / len(self.action_history)
            self.restlessness = 0.9 * self.restlessness + 0.1 * oscillation_rate
        
        # --- Urgency (Time Pressure) ---
        # If allowed to go (decision=0) but speed is low
        decision = getattr(env, 'decision', 0)
        pedestrian_ahead = getattr(env, 'pedestrian_ahead', False)
        
        if decision == 0 and not pedestrian_ahead and speed < 5.0:
            self.urgency = min(self.urgency + 0.05, 1.0) # Grow over time
        else:
            self.urgency = max(self.urgency - 0.1, 0.0) # Decay when moving or stopped correctly
            
        # --- Frustration (Cumulative Negative/Low Reward) ---
        avg_reward = np.mean(self.reward_history) if self.reward_history else 0
        if avg_reward < 0:
            self.frustration = min(self.frustration + 0.05, 1.0)
        else:
            self.frustration = max(self.frustration - 0.05, 0.0)

    def compute_reward(self):
        # Weighted sum of emotions
        r_calm = self.calmness * self.w_calm
        r_rest = self.restlessness * self.w_restless
        r_urg = self.urgency * self.w_urgency
        r_frust = self.frustration * self.w_frustion
        
        total_emotion_reward = (r_calm + r_rest + r_urg + r_frust) * self.global_scale
        return total_emotion_reward
        
    def get_info(self):
        return {
            "Action/Calmness": self.calmness,
            "Action/Restlessness": self.restlessness,
            "Action/Urgency": self.urgency,
            "Action/Frustration": self.frustration,
            "Reward/Emotion": self.compute_reward()
        }


def traffic_light_reward_fn(env):
    """
    Reward function for Decision-Based Hybrid Control:
    - RED + STOP (Decision 1) -> + Reward
    - RED + GO (Decision 0)   -> - Penalty
    - GREEN + GO (Decision 0) -> + Reward
    - GREEN + STOP (Decision 1) -> - Penalty
    - Outside Zone -> 0 (Or small living reward)
    
    Integrated with Emotion-Based Rewards.
    """
    
    # Initialize Emotion State if not present
    if not hasattr(env, 'emotion_state'):
        env.emotion_state = EmotionState()

    tl_state = env.tl_state # 0: Red, 1: Yellow, 2: Green
    tl_dist = env.tl_dist
    decision = getattr(env, 'decision', 0) # 0=GO, 1=STOP
    
    reward = 0.0
    
    # 1. Zone Check (Reward active only inside zone)
    ZONE_THRESHOLD = 30.0
    
    is_in_zone = (tl_dist <= ZONE_THRESHOLD)
    
    if not is_in_zone:
        reward = 0.0 # Neutral outside zone (Rule-based handled)
    else:
        # Inside Zone
        if tl_state == 2: # GREEN
            if decision == 0: # GO
                reward += 5.0 # Strong positive for moving
            else: # STOP
                reward -= 5.0 # Strong penalty for blocking
                
        elif tl_state == 0 or tl_state == 1: # RED / YELLOW
            # Simplification: Yellow = Red if in decision zone
            
            if decision == 1: # STOP
                reward += 5.0 # Strong positive for stopping
            else: # GO
                reward -= 10.0 # Stronger penalty for running red
                
                # Check for actual violation (crossing line)
                if tl_dist < 5.0:
                     reward -= 50.0 # Critical violation

    # Collision (handled by wrapper usually, but good to reinforce)
    if env.collision_detected:
         reward -= 50.0

    # --- PEDESTRIAN REWARDS (MASKED) ---
    pedestrian_ahead = getattr(env, 'pedestrian_ahead', False)
    
    if pedestrian_ahead:
        # Overwrite/Add to traffic light reward?
        # Safety takes priority.
        
        ped_reward = 0.0
        ped_dist = getattr(env, 'ped_dist', 100.0)
        
        if decision == 1: # STOP
            ped_reward += 5.0 # Strong positive for stopping
        else: # GO
            # Penalty scales with distance
            if ped_dist < 15.0:
                ped_reward -= 20.0 # Extreme danger
            else:
                ped_reward -= 5.0 # Warning
        
        # Speed penalty if very close
        current_speed = env.vehicle.get_speed()
        if ped_dist < 20.0 and current_speed > 5.0:
            ped_reward -= (current_speed / 5.0) # Penalty for moving fast
            
        reward += ped_reward
        
    # --- EMOTION INTEGRATION ---
    base_reward = reward
    
    # Update emotion state
    env.emotion_state.update(env, base_reward)
    emotion_reward = env.emotion_state.compute_reward()
    
    # Total Reward
    total_reward = base_reward + emotion_reward
    
    # Log Emotions
    emo_info = env.emotion_state.get_info()
    for k, v in emo_info.items():
        # Clean naming for HUD/Logs
        env.extra_info.append(f"{k}: {v:.2f}")

    return total_reward


# Register the reward function
reward_functions["traffic_light_reward"] = create_reward_fn(traffic_light_reward_fn)
# Also map to reward_fn5 for compatibility if config demands, or we change config

