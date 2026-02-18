import warnings
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import config
import time

config.set_config("1")

from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from carla_env.carla_route_env import CarlaRouteEnv

from vae.utils.misc import LSIZE
from carla_env.state_commons import create_encode_state_fn, load_vae
from carla_env.rewards import reward_functions
from utils import HParamCallback, TensorboardCallback, write_json, parse_wrapper_class

from config import CONFIG

log_dir = 'tensorboard'
os.makedirs(log_dir, exist_ok=True)
reload_model = ""
total_timesteps = 10000000

seed = CONFIG["seed"]

AlgorithmRL = CONFIG["algorithm"]
vae = None
if CONFIG["vae_model"]:
    vae = load_vae(f'./vae/log_dir/{CONFIG["vae_model"]}', LSIZE)
observation_space, encode_state_fn, decode_vae_fn = create_encode_state_fn(vae, CONFIG["state"])

env = CarlaRouteEnv(obs_res=CONFIG["obs_res"], host="localhost", port=2000,
                    reward_fn=reward_functions[CONFIG["reward_fn"]],
                    observation_space=observation_space,
                    encode_state_fn=encode_state_fn, decode_vae_fn=decode_vae_fn,
                    fps=15, action_smoothing=CONFIG["action_smoothing"],
                    action_space_type='continuous', activate_spectator=False, activate_render=True) #change render

for wrapper_class_str in CONFIG["wrappers"]:
    wrap_class, wrap_params = parse_wrapper_class(wrapper_class_str)
    env = wrap_class(env, *wrap_params)

if reload_model == "":
    model = AlgorithmRL('MultiInputPolicy', env, verbose=1, seed=seed, tensorboard_log=log_dir, device='cuda',
                        **CONFIG["algorithm_params"])
    model_suffix = f"{int(time.time())}_DDPG"
else:
    model = AlgorithmRL.load(reload_model, env=env, device='cuda', seed=seed, **CONFIG["algorithm_params"])
    model_suffix = f"{reload_model.split('/')[-2].split('_')[-1]}_finetuning"

model_name = f'{model.__class__.__name__}_{model_suffix}'

model_dir = os.path.join(log_dir, model_name)
new_logger = configure(model_dir, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)
write_json(CONFIG, os.path.join(model_dir, 'config.json'))

model.learn(total_timesteps=total_timesteps,
            callback=[HParamCallback(CONFIG), TensorboardCallback(1), CheckpointCallback(
                save_freq=total_timesteps // 10,
                save_path=model_dir,
                name_prefix="model")], reset_num_timesteps=False)

