# source/isaaclab_assets/grippers/robotiq_2f140_cfg.py
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

ROBOTIQ_2F140_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/HDD/_projects/github/robot/data/descriptions/robotiq/Robotiq_2F_140_physics_edit.usd",  
        activate_contact_sensors=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Actuated master joint (others are mimic):
            "left_inner_finger_joint": 0.0,
        },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            # Use a permissive pattern to catch names with prefixes (e.g., when attached under a tool frame).
            joint_names_expr=[
                # ".*finger_joint$",
                # # Common explicit names if patterns are restricted:
                # "finger_joint", "left_inner_finger_joint", "right_inner_finger_joint",
                # "left_finger_joint", "right_finger_joint",
                "left_inner_finger_joint",
            ],
            # From the xacro: effort="1000" on finger-related joints. You can lower this if needed.
            effort_limit_sim=1000.0,
            # Grippers usually benefit from higher gains for snappy open/close:
            stiffness=2000.0,
            damping=100.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# Higher PD variant (kept identical here; adjust if you need even stiffer behavior)
ROBOTIQ_2F140_HIGH_PD_CFG = ROBOTIQ_2F140_CFG.copy()
ROBOTIQ_2F140_HIGH_PD_CFG.actuators["fingers"].stiffness = 3000.0
ROBOTIQ_2F140_HIGH_PD_CFG.actuators["fingers"].damping = 150.0

# Helper: nominal joint bounds for mapping (for task configs).
ROBOTIQ_2F140_JOINT_NAME = "finger_joint"
ROBOTIQ_2F140_JOINT_LIMITS = (0.0, 0.7)  # radians, from xacro <limit lower="0" upper="0.7">
