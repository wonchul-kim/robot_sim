"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""
import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# from pathlib import Path 
# FILE = Path(__file__).resolve()
# ROOT = FILE.parent

M1013_2F140_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/HDD/_projects/github/robot/data/descriptions/doosan/m1013_2f140.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=1
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={'joint_1': 0.0, 
                   'joint_2': -math.radians(10.0),
                   'joint_3': math.radians(90.0),
                   'joint_4': 0.0,
                   'joint_5': math.radians(90.0),
                   'joint_6': 0.0,
                   'robotiq_hande_left_finger_joint': 0.01,
                   'robotiq_hande_right_finger_joint': 0.01},
    ),
    actuators={
        "m1013_group1": ImplicitActuatorCfg(
            joint_names_expr=['joint_1', 'joint_2'],
            effort_limit_sim=20400,
            stiffness=566.6667,
            damping=56.6667,
        ),
        "m1013_group2": ImplicitActuatorCfg(
            joint_names_expr=['joint_3'],
            effort_limit_sim=9600,
            stiffness=266.6667,
            damping=26.6667,
        ),
        "m1013_group3": ImplicitActuatorCfg(
            joint_names_expr=['joint_4', 'joint_5', 'joint_6'],
            effort_limit_sim=2700,
            stiffness=75,
            damping=7.5,
        ),
        "hande_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "robotiq_hande_left_finger_joint",
                "robotiq_hande_right_finger_joint",
            ],
            # Isaac 내부 단위가 N (prismatic)로 매핑되므로 아래를 적당히 보수적으로 설정
            effort_limit_sim=1300.0,
            stiffness=3000.0,   # m-DOF라 회전보다 큰 값이 자연스러움
            damping=50.0,
        ),
    },
    # soft_joint_pos_limit_factor=1.0,
)


# """Configuration of Franka Emika Panda robot with stiffer PD control.

# This configuration is useful for task-space control using differential IK.
# """

