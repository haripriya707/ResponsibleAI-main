import os
import argparse
import pandas as pd
import numpy as np
import config
import re

parser = argparse.ArgumentParser(description="Eval a CARLA agent")
parser.add_argument("--host", default="localhost", type=str, help="IP of the host server (default: 127.0.0.1)")
parser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")
parser.add_argument("--model", type=str, default="", help="Path to a model evaluate")
parser.add_argument("--no_render", action="store_false", help="If True, render the environment")
parser.add_argument("--fps", type=int, default=15, help="FPS to render the environment")
parser.add_argument("--no_record_video", action="store_false", help="If True, record video of the evaluation")
parser.add_argument("--config", type=str, default="1", help="Config to use (default: 1)")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")

args = vars(parser.parse_args())
config.set_config(args["config"])

from stable_baselines3 import PPO, DDPG, SAC

from utils import VideoRecorder, parse_wrapper_class
from carla_env.state_commons import create_encode_state_fn, load_vae
from carla_env.rewards import reward_functions

from vae.utils.misc import LSIZE
from carla_env.wrappers import vector, get_displacement_vector
from carla_env.carla_route_env import CarlaRouteEnv
from eval_plots import plot_eval, summary_eval

from config import CONFIG


def get_model_path_from_train():
    """Extracts reload_model path from train.py"""
    try:
        with open("train.py", "r") as f:
            content = f.read()
            match = re.search(r'reload_model\s*=\s*"([^"]+)"', content)
            if match:
                return match.group(1)
    except Exception as e:
        print(f"Error reading train.py: {e}")
    return ""

def run_eval(env, model, model_path=None, record_video=False, num_episodes=5):
    model_name = os.path.basename(model_path)
    log_path = os.path.join(os.path.dirname(model_path), 'eval')
    os.makedirs(log_path, exist_ok=True)
    video_path = os.path.join(log_path, model_name.replace(".zip", "_eval.avi"))
    csv_path = os.path.join(log_path, model_name.replace(".zip", "_eval.csv"))
    model_id = f"{model_path.split('/')[-2]}-{model_name.split('_')[-2]}"
    
    # Init video recording
    if record_video:
        rendered_frame = env.render(mode="rgb_array")
        print("Recording video to {} ({}x{}x{}@{}fps)".format(video_path, *rendered_frame.shape,
                                                              int(env.fps)))
        video_recorder = VideoRecorder(video_path,
                                       frame_size=rendered_frame.shape,
                                       fps=env.fps)
    else:
        video_recorder = None

    columns = ["model_id", "episode", "step", "throttle", "steer", "brake", "vehicle_location_x", "vehicle_location_y",
               "reward", "distance", "speed", "center_dev", "angle_next_waypoint", "waypoint_x", "waypoint_y",
               "route_x", "route_y", "in_zone", "pedestrian_ahead", "tl_state", "tl_dist", "decision"]
    df = pd.DataFrame(columns=columns)

    # Metrics Storage
    metrics = {
        "episodes": [],
        "ped_braking_success": 0,
        "total_ped_encounters": 0,
        "tl_compliance_count": 0,
        "total_tl_encounters": 0, # Red light encounters
        "collisions": 0,
        "total_steps": 0,
        "total_reward": 0
    }

    print(f"\nStarting Evaluation for {num_episodes} episodes...")

    for episode_idx in range(num_episodes):
        print(f"Episode {episode_idx + 1}/{num_episodes}")
        state, info = env.reset() # Assuming env.reset returns (obs, info) per carla_route_env.py
        
        # Handle tuple return if gym wrapper interferes, though carla_route_env returns (obs, info)
        if isinstance(state, tuple):
             state, info = state
        
        done = False
        episode_reward = 0
        episode_steps = 0
        
        # Episode Specific Metrics
        ep_metrics = {
            "stops": 0,
            "red_light_violations": 0,
            "ped_encounters": 0,
            "ped_collisions": 0,
            "ped_braking_events": 0,
            "brake_while_ped": False,
            "ped_seen_this_ep": False
        }

        saved_route = False

        while not done:
            # Context Gating Logic
            tl_dist = info.get('tl_dist', 100.0)
            pedestrian_ahead = info.get('pedestrian_ahead', False)
            tl_state = info.get('tl_state', 2) # 0=Red, 1=Yellow, 2=Green
            
            in_zone = (tl_dist < 30.0) # Traffic Light Zone
            if pedestrian_ahead:
                in_zone = True
            
            # Action Selection
            if in_zone:
                action, _states = model.predict(state, deterministic=True)
            else:
                action = 0 # Force GO (0)

            # Step Environment
            # carla_route_env returns 5 items
            state, reward, done, truncated, info = env.step(action)
            # If info is new, update local vars for next loop if needed, but we check 'info' at start of loop usually.
            # Wait, logic check: 'info' is from *previous* step (or reset) when entering logic. 
            # We used info from reset() for first step. Correct.
            
            episode_reward += reward
            episode_steps += 1
            metrics["total_steps"] += 1
            
            # Use info from CURRENT step for metrics logging (post-action state)
            
            # Pedestrians
            if info.get('pedestrian_ahead', False):
                if not ep_metrics["ped_seen_this_ep"]:
                    ep_metrics["ped_encounters"] += 1
                    ep_metrics["ped_seen_this_ep"] = True 
                    metrics["total_ped_encounters"] += 1
                
                # Check for braking
                # Checking action == 1 (STOP) or actual brake control
                if action == 1 or env.vehicle.control.brake > 0.1:
                    ep_metrics["brake_while_ped"] = True
                    ep_metrics["ped_braking_events"] += 1

            else:
                ep_metrics["ped_seen_this_ep"] = False 

            # Traffic Lights
            if tl_state == 0: # Red Light (from previous info, which decided action)
                 if in_zone:
                     # Check compliance
                     if env.vehicle.get_speed() < 1.0: # stopped
                         ep_metrics["stops"] += 1
                     
                     # Simple Violation Check: Moving fast while red and close
                     if tl_dist < 10.0 and action == 0 and env.vehicle.get_speed() > 5.0:
                         ep_metrics["red_light_violations"] += 1

            # Collisions
            if env.collision_detected:
                 if getattr(env, 'obst_type', '') == "pedestrian":
                     ep_metrics["ped_collisions"] += 1
                 metrics["collisions"] += 1

            # --- LOGGING --- 
            if not saved_route:
                initial_heading = np.deg2rad(env.vehicle.get_transform().rotation.yaw)
                initial_vehicle_location = vector(env.vehicle.get_location())
                for way in env.route_waypoints:
                    route_relative = get_displacement_vector(initial_vehicle_location,
                                                             vector(way[0].transform.location),
                                                             initial_heading)
                    new_row = pd.DataFrame([['route', episode_idx, route_relative[0], route_relative[1]]],
                                           columns=["model_id", "episode", "route_x", "route_y"])
                    df = pd.concat([df, new_row], ignore_index=True)
                saved_route = True

            vehicle_relative = get_displacement_vector(initial_vehicle_location, vector(env.vehicle.get_location()),
                                                    initial_heading)
            waypoint_relative = get_displacement_vector(initial_vehicle_location,
                                                        vector(env.current_waypoint.transform.location), initial_heading)

            new_row = pd.DataFrame(
                [[model_id, episode_idx, episode_steps, 
                  env.vehicle.control.throttle, env.vehicle.control.steer, env.vehicle.control.brake,
                  vehicle_relative[0], vehicle_relative[1], reward,
                  env.distance_traveled,
                  env.vehicle.get_speed(), env.distance_from_center,
                  np.rad2deg(env.vehicle.get_angle(env.current_waypoint)),
                  waypoint_relative[0], waypoint_relative[1], None, None,
                  in_zone, pedestrian_ahead, tl_state, tl_dist, action
                  ]], columns=columns)
            df = pd.concat([df, new_row], ignore_index=True)

            if record_video and video_recorder:
                video_recorder.add_frame(env.render(mode="rgb_array"))

            if done or truncated:
                break

        # End of Episode Stats
        print(f"  > Steps: {episode_steps}, Reward: {episode_reward:.2f}")
        print(f"  > Ped Encounters: {ep_metrics['ped_encounters']}, Braking Events: {ep_metrics['ped_braking_events']}")
        print(f"  > Traffic Stops: {ep_metrics['stops']}, Violations: {ep_metrics['red_light_violations']}")
        print(f"  > Ped Collisions: {ep_metrics['ped_collisions']}")
        
        if ep_metrics['ped_encounters'] > 0 and ep_metrics['brake_while_ped']:
             metrics["ped_braking_success"] += 1
        
        metrics["episodes"].append(ep_metrics)
        metrics["total_reward"] += episode_reward

    # -- Final Summary --
    print("\n" + "="*40)
    print("EVALUATION SUMMARY")
    print("="*40)
    print(f"Total Episodes: {num_episodes}")
    print(f"Avg Reward: {metrics['total_reward']/num_episodes:.2f}")
    print(f"Total Collisions: {metrics['collisions']}")
    
    total_peds = metrics['total_ped_encounters']
    success_rate = (metrics['ped_braking_success'] / total_peds * 100) if total_peds > 0 else 0.0
    print(f"Pedestrian Encounters: {total_peds}")
    print(f"Pedestrian Braking Success Rate: {success_rate:.1f}%")
    
    total_violations = sum(e['red_light_violations'] for e in metrics['episodes'])
    print(f"Red Light Violations: {total_violations}")

    if record_video and video_recorder:
        video_recorder.release()

    df.to_csv(csv_path, index=False)
    try:
        plot_eval([csv_path])
        summary_eval(csv_path)
    except Exception as e:
        print(f"Plotting failed or skipped: {e}")


if __name__ == "__main__":
    if args["model"] == "":
        model_path = get_model_path_from_train()
        if model_path == "":
            raise ValueError("No model path provided and could not extract from train.py")
        print(f"Using model from train.py: {model_path}")
    else:
        model_path = args["model"]

    val = CONFIG["algorithm"]
    if isinstance(val, str):
        algorithm_dict = {"PPO": PPO, "DDPG": DDPG, "SAC": SAC}
        if val not in algorithm_dict:
            raise ValueError(f"Invalid algorithm name: {val}")
        AlgorithmRL = algorithm_dict[val]
    else:
        AlgorithmRL = val

    vae = None
    if CONFIG["vae_model"]:
        vae = load_vae(f'./vae/log_dir/{CONFIG["vae_model"]}', LSIZE)
    
    observation_space, encode_state_fn, decode_vae_fn = create_encode_state_fn(vae, CONFIG["state"])

    # Initialize Env
    # Specifying eval=True
    env = CarlaRouteEnv(obs_res=CONFIG["obs_res"], viewer_res=(1120, 560), host=args["host"], port=args["port"],
                        reward_fn=reward_functions[CONFIG["reward_fn"]],
                        observation_space=observation_space,
                        encode_state_fn=encode_state_fn, decode_vae_fn=decode_vae_fn,
                        fps=args["fps"], action_smoothing=CONFIG["action_smoothing"],
                        action_space_type='continuous', activate_spectator=True, eval=True,
                        activate_render=args["no_render"], start_carla=True) 

    # Wrappers
    for wrapper_class_str in CONFIG["wrappers"]:
        wrap_class, wrap_params = parse_wrapper_class(wrapper_class_str)
        env = wrap_class(env, *wrap_params)

    # Load Model
    model = AlgorithmRL.load(model_path, env=env, device='cuda')

    # Run Eval
    run_eval(env, model, model_path, record_video=args['no_record_video'], num_episodes=args['num_episodes'])

