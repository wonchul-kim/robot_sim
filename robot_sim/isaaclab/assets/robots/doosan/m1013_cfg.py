"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# from pathlib import Path 
# FILE = Path(__file__).resolve()
# ROOT = FILE.parent

M1013_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/HDD/_projects/github/robot/data/descriptions/doosan/m1013.usd",
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
                   'joint_2': 0.0,
                   'joint_3': 0.0,
                   'joint_4': 0.0,
                   'joint_5': 0.0,
                   'joint_6': 0.0},
    ),
    actuators={
        "m1013_group1": ImplicitActuatorCfg(
            joint_names_expr=['joint_1', 'joint_2', 'joint_3'],
            effort_limit_sim=30.000,
            stiffness=80.0,
            damping=4.0,
        ),
        "m1013_group2": ImplicitActuatorCfg(
            joint_names_expr=['joint_4', 'joint_5', 'joint_6'],
            effort_limit_sim=30.000,
            stiffness=80.0,
            damping=4.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


# M1013_HIGH_PD_CFG = M1013_CFG.copy()
# M1013_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
# M1013_HIGH_PD_CFG.actuators["m1013_group1"].stiffness = 400.0
# M1013_HIGH_PD_CFG.actuators["m1013_group1"].damping = 80.0
# M1013_HIGH_PD_CFG.actuators["m1013_group2"].stiffness = 400.0
# M1013_HIGH_PD_CFG.actuators["m1013_group2"].damping = 80.0

# """Configuration of Franka Emika Panda robot with stiffer PD control.

# This configuration is useful for task-space control using differential IK.
# """

# # Note:
# # - Use "tool0" as the end-effector frame in your DifferentialIKController.
# # - Base link for kinematics is "base_link".
