import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
from isaaclab.managers import EventTermCfg  

from .m1013_cfg import M1013_CFG
from robot_sim.isaaclab.assets.robots.ur.ur10 import UR10_CFG
from robot_sim.isaaclab.assets.robots.robotiq.robotiq_2f140_cfg import ROBOTIQ_2F140_CFG


# @configclass
# class M1013ReachEnvCfg(ReachEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()

#         # switch robot to ur10
#         self.scene.robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
#         # override events
#         self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
#         # override rewards
#         self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["ee_link"]
#         self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["ee_link"]
#         self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["ee_link"]
#         # override actions
#         self.actions.arm_action = mdp.JointPositionActionCfg(
#             asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
#         )
#         # override command generator body
#         # end-effector is along x-direction
#         self.commands.ee_pose.body_name = "ee_link"
#         self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)


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

@configclass
class M1013Robotiq2F140ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur10
        self.scene.robot = M1013_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.gripper = ROBOTIQ_2F140_CFG.replace(prim_path="{ENV_REGEX_NS}/Gripper")

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

        # self.actions.gripper_action = mdp.JointPositionActionCfg(
        #     asset_name="gripper", joint_names=[r".*finger_joint$"],      # 그리퍼 1DoF
        #     scale=1.0, use_default_offset=False
        # )
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="gripper",
            joint_names=["left_inner_finger_joint"],  # ← 실존 이름
            scale=1.0,
            use_default_offset=False
        )

        # 4) (중요) 초기화 이벤트에 attach 절차를 추가
        #    사용 중인 Isaac Lab 버전에 맞는 attach 이벤트/헬퍼를 등록하세요.
        #    예시: reset 시점에 'gripper'를 'robot'의 tool0 링크에 고정 결합.
        self.events.attach_gripper = EventTermCfg(
            func="robot_sim.isaaclab.assets.robots.doosan.attachments:attach_articulation_fixed",
            mode="on_reset",
            params=dict(
                parent_asset="robot",
                parent_link="tool0",
                child_asset="gripper",
                pose_in_parent=dict(p=[0.0, 0.0, 0.0], rpy=[0.0, 0.0, 0.0]),  # 필요 시 장착 오프셋 조정
                fixed_joint=True,
            ),
        )
