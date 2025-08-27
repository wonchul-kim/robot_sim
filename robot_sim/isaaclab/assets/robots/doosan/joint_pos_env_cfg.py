import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

from .m1013_cfg import M1013_CFG



@configclass
class M1013ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur10
        self.scene.robot = M1013_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # # override events
        # self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["tool0"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["tool0"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["tool0"]
        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "tool0"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)
