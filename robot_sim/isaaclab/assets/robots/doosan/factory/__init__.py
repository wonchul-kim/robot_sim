import gymnasium as gym

from . import agents
from .factory_env import FactoryEnv
from .factory_env_cfg import FactoryTaskGearMeshCfg, FactoryTaskNutThreadCfg, FactoryTaskPegInsertCfg


gym.register(
    id="Isaac-Factory-PegInsert-Direct-M1013-2F140-v0",
    entry_point="isaaclab_tasks.direct.factory:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)