

"""Launch Isaac Sim Simulator first."""

import argparse 
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description='')
parser.add_argument("--output_dir", default='/HDD/etc/outputs/isaac', type=str)
parser.add_argument("--device", default='cuda:1', type=str)
parser.add_argument("--gui", action="store_true", default=False)
parser.add_argument("--video", action="store_true", default=True, help="Record videos during training")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=20000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default='Isaac-Reach-Franka-v0', help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=1e6, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)

cli_args = parser.parse_args()
'''
cli_args: Namespace(headless=False, livestream=-1, 
                    enable_cameras=False, xr=False, device='cuda:0', 
                    cpu=False, verbose=False, info=False, experience='', 
                    rendering_mode=None, kit_args='')
'''
if cli_args.gui:
    cli_args.headless = True
if cli_args.video:
    cli_args.enable_cameras = True
print(f"cli_args: {cli_args}")

### launch omniverse app
app_launcher = AppLauncher(cli_args)
simulation_app = app_launcher.app 

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext

import os
import torch 
from datetime import datetime 

import gymnasium as gym
from isaaclab.utils.dict import print_dict 
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

import isaaclab_tasks
from isaaclab_tasks.utils import get_checkpoint_path 
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

from robot_sim.utils.logger import Logger

@hydra_task_config(cli_args.task, '')
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg=None):
    
    """Train with RSL-RL agent."""
    ### override configurations with non-hydra CLI arguments
    # agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, cli_args)
    # agent_cfg.max_iterations = (
    #     cli_args.max_iterations if cli_args.max_iterations is not None else agent_cfg.max_iterations
    # )
    env_cfg.scene.num_envs = cli_args.num_envs if cli_args.num_envs is not None else env_cfg.scene.num_envs

    ### set the environment seed
    ### note: certain randomizations occur in the environment initialization so we set the seed here
    # env_cfg.seed = agent_cfg.seed
    env_cfg.seed = 42
    env_cfg.sim.device = cli_args.device if cli_args.device is not None else env_cfg.sim.device

    ### multi-gpu training configuration
    if cli_args.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        # agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        ### set seed to have diversity in different threads
        # seed = agent_cfg.seed + app_launcher.local_rank
        seed = 42 + app_launcher.local_rank
        env_cfg.seed = seed
        # agent_cfg.seed = seed

    ### specify directory for logging experiments
    log_root_path = os.path.join(cli_args.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    log_dir = os.path.join(log_root_path, 'logs')
    logger = Logger(name='tmp', 
                    log_dir=log_dir, 
                    log_file_level="DEBUG")
    logger.info(f"[INFO] Logging experiment in directory: {log_dir}")

    ### create isaac environment ------------------------------------------------------------------
    env = gym.make(cli_args.task, cfg=env_cfg, render_mode="rgb_array" if cli_args.video else None)

    ### convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    ### save resume path before creating a new log_dir
    # if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
    #     resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if cli_args.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, "videos", "train"),
            "step_trigger": lambda step: step % cli_args.video_interval == 0,
            "video_length": cli_args.video_length,
            "disable_logger": True,
        }
        logger.info(f"[INFO] Recording videos during training:")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ### wrap around environment for rsl-rl
    # env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ### create runner from rsl-rl
    # runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # ### write git state to logs
    # runner.add_git_repo_to_log(__file__)
    # ## load the checkpoint
    # if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
    #     print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    #     # load previously trained model
    #     runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    # dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    ### run training
    # runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == '__main__':

    main()
    
    simulation_app.close()
    