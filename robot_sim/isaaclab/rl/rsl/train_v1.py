if __name__ == '__main__':
    
    import argparse
    import cli_args as cli_args_  # isort: skip
    from robot_sim.isaaclab.envs.env import get_env

    simulation_app, app_launcher, env, cli_args = get_env()
    
    rls_args = cli_args_.get_rsl_rl_args()
    cli_args = argparse.Namespace(**{**vars(rls_args), **vars(cli_args)})

    print(env)
    import os
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab.utils.io import dump_pickle, dump_yaml
    from isaaclab_tasks.utils import get_checkpoint_path 
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings
    
    agent_cfg_entry_point = 'rsl_rl_cfg_entry_point'
    agent_cfg = None
    agent_cfg = load_cfg_from_registry(cli_args.task.split(":")[-1], agent_cfg_entry_point)
    
    
    """Train with RSL-RL agent."""
    ### override configurations with non-hydra CLI arguments
    agent_cfg = cli_args_.update_rsl_rl_cfg(agent_cfg, cli_args)
    agent_cfg.max_iterations = (
        cli_args.max_iterations if cli_args.max_iterations is not None else agent_cfg.max_iterations
    )
    
    ### multi-gpu training configuration
    if cli_args.distributed:
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        ### set seed to have diversity in different threads
        seed = cli_args.seed + app_launcher.local_rank
        agent_cfg.seed = seed
    
    ### save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(cli_args.log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    ### wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    ### create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=cli_args.log_dir, device=agent_cfg.device)
    ### write git state to logs
    runner.add_git_repo_to_log(__file__)
    ## load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    ### dump the configuration into log-directory
    dump_yaml(os.path.join(cli_args.log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(cli_args.log_dir, "params", "agent.pkl"), agent_cfg)

    ### run training
    runner.learn(num_learning_iterations=int(agent_cfg.max_iterations), init_at_random_ep_len=True)

    env.close()

    simulation_app.close()
    