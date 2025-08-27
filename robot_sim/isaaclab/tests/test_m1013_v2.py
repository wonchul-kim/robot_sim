import argparse
import torch
import math


### 1. Launch Omniverse =====================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

### 2. Scene (ground + light + robot) =====================================================================
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
# sim_utils.DomeLightCfg(intensity=1000.0).func("/World/Light", sim_utils.DomeLightCfg())

sim_utils.DomeLightCfg(
    intensity=30000.0,   # 필요시 1e5~3e5까지 올려보세요
    texture_file=None,   # HDRI 쓰면 더 자연스럽게
    exposure=0.0
).func("/World/EnvLight", sim_utils.DomeLightCfg())

# origin xform
prim_utils.create_prim("/World/Origin", "Xform", translation=(0.0, 0.0, 0.0))

### 3. Define robot =====================================================================
from robot_sim.isaaclab.assets.robots.doosan.m1013_cfg import M1013_CFG
from robot_sim.isaaclab.assets.robots.franka.franka import FRANKA_PANDA_CFG
from robot_sim.isaaclab.assets.robots.ur.ur10 import UR10_CFG

cfg = M1013_CFG.copy()
cfg.prim_path = "/World/Origin/Robot"

from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation

robot = Articulation(cfg=cfg)

# 4. Simulation =====================================================================
sim = SimulationContext(sim_utils.SimulationCfg())
sim.reset()

# On reset: write default root & joint state, then reset buffers
root_state = robot.data.default_root_state.clone()
print("ROOT STATE: ", root_state)
robot.write_root_pose_to_sim(root_state[:, :7])
robot.write_root_velocity_to_sim(root_state[:, 7:])
robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
robot.reset()

print("== Joint names ==")
print(robot.joint_names)
num_joints = len(robot.joint_names)

print("== DOF indices ==")
print(robot.data.joint_pos.shape)   # [num_envs, num_dofs]

print("== Link  names ==")
import omni.usd
stage = omni.usd.get_context().get_stage()
for prim in stage.Traverse():
    if prim.IsA(omni.usd.UsdGeom.Xform):  # 또는 UsdGeom.Mesh
        print(prim.GetPath().pathString)


# Base Coordinates
print('root_pos_w: ', robot.data.root_pos_w)      # [N, 3]  월드 좌표계에서 베이스 위치
print('root_quat_w: ', robot.data.root_quat_w)     # [N, 4]  월드 좌표계에서 베이스 회전
print('root_lin_vel_w: ', robot.data.root_lin_vel_w)  # [N, 3]  월드 좌표계 선속도
print('root_ang_vel_w: ', robot.data.root_ang_vel_w)  # [N, 3]  월드 좌표계 각속도
    
# 시작 각도: cfg.init_state에 준 값으로 시작
q0 = robot.data.default_joint_pos.clone().cpu().detach()  # shape: [1, num_joints]
print(f'Current Joint Pos: {robot.data.joint_pos}')
qd0 = torch.zeros_like(q0)

# 간단한 사인 궤적을 3축(혹은 모든 축)에 줘서 움직임 확인
# 각도 범위가 불명확하니 작은 진폭(라디안)으로만 테스트
amp = torch.tensor([[0.2]*num_joints])  # 0.2rad ≈ 11.5deg
freq = 0.2  # Hz
dt = sim.get_physics_dt() if hasattr(sim, "get_physics_dt") else 1.0/60.0
t = 0.0

import numpy as np
from robot_sim.isaaclab.utils.articulations import get_ee_state_world

trajectory = {'joint_position': [], 'joint_velocity': [], 'ee_pos_w': [], 'ee_quat_w_xyzw': [], 'ee_lin_w': [], 'ee_ang_w': []}

sim.play()
steps = 500  # 약 5초 @60Hz
for i in range(steps): ###  =====================================================================
    t += dt 
    # 목표 각도(부드럽게 왔다갔다)
    q_target = q0 + amp * torch.sin(torch.tensor([[2*math.pi*freq*t]]))  # [1, dof]
    qd_target = torch.zeros_like(q_target)

    ee = get_ee_state_world(robot, ee_name="tool0", env_idx=0, to_numpy=True)

    trajectory['joint_position'].append(robot.data.joint_pos.cpu().detach().numpy().tolist()[0])
    trajectory['joint_velocity'].append(robot.data.joint_vel.cpu().detach().numpy().tolist()[0])
    trajectory['ee_pos_w'].append(ee['pos_w'].tolist())
    trajectory['ee_quat_w_xyzw'].append(ee['quat_w_xyzw'].tolist())
    trajectory['ee_lin_w'].append(ee['lin_w'].tolist())
    trajectory['ee_ang_w'].append(ee['ang_w'].tolist())
    
    if i%200 == 0:
        print(f'q_target: {q_target}')
        print(f'robot.data.joint_pos: {robot.data.joint_pos}')
        print(f'robot.data.joint_vel: {robot.data.joint_vel}')

        # # 사용 예:
        # print("[EE] index:", ee["index"], "name_field:", ee["name_field"], "tensors:", ee["tensor_fields"])
        print("pos_w :", ee["pos_w"])
        print("quat_w:", ee["quat_w_xyzw"])
        print("lin_w :", ee["lin_w"])
        print("ang_w :", ee["ang_w"])

    # “움직이는지 확인” 용도로 매 스텝 joint state를 조금씩 덮어쓰기
    robot.write_joint_state_to_sim(q_target, qd_target)

    # 한 스텝 진행
    sim.step()
    
import pandas as pd 
df = pd.DataFrame(trajectory)
df.to_csv('/HDD/etc/outputs/isaac/trajectory.csv')

# 종료 정리
sim.stop()
