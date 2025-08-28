from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
from robot_sim.isaaclab.assets.robots.doosan.m1013_2f140_cfg import M1013_2F140_CFG

'''
== Link names == 
/World/ground False
/World/ground/GroundPlane False
/World/ground/Environment False
/World/Origin False
/World/Origin/Robot False
/World/Origin/Robot/base True
/World/Origin/Robot/base/base_link False
/World/Origin/Robot/base/visuals False
/World/Origin/Robot/base/collisions False
/World/Origin/Robot/link_1 True
/World/Origin/Robot/link_1/visuals False
/World/Origin/Robot/link_1/collisions False
/World/Origin/Robot/link_2 True
/World/Origin/Robot/link_2/visuals False
/World/Origin/Robot/link_2/collisions False
/World/Origin/Robot/link_3 True
/World/Origin/Robot/link_3/visuals False
/World/Origin/Robot/link_3/collisions False
/World/Origin/Robot/link_4 True
/World/Origin/Robot/link_4/visuals False
/World/Origin/Robot/link_4/collisions False
/World/Origin/Robot/link_5 True
/World/Origin/Robot/link_5/visuals False
/World/Origin/Robot/link_5/collisions False
/World/Origin/Robot/link_6 True
/World/Origin/Robot/link_6/tool0 False
/World/Origin/Robot/link_6/tool0/robotiq_hande_coupler False
/World/Origin/Robot/link_6/tool0/robotiq_hande_coupler/robotiq_hande_link False
/World/Origin/Robot/link_6/tool0/robotiq_hande_coupler/robotiq_hande_link/robotiq_hande_end False
/World/Origin/Robot/link_6/visuals False
/World/Origin/Robot/link_6/collisions False
/World/Origin/Robot/robotiq_hande_left_finger True
/World/Origin/Robot/robotiq_hande_left_finger/visuals False
/World/Origin/Robot/robotiq_hande_left_finger/collisions False
/World/Origin/Robot/robotiq_hande_right_finger True
/World/Origin/Robot/robotiq_hande_right_finger/visuals False
/World/Origin/Robot/robotiq_hande_right_finger/collisions False
'''

@configclass
class M10132F140CubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = M1013_2F140_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["robotiq_hande_left_finger_joint", "robotiq_hande_right_finger_joint"],
            open_command_expr={"robotiq_hande_left_finger_joint": 0.025,
                               "robotiq_hande_right_finger_joint": 0.025},
            close_command_expr={"robotiq_hande_left_finger_joint": 0.,
                                "robotiq_hande_right_finger_joint": 0.},
        )

        self.commands.object_pose.body_name = "link_6"

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/link_6",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )