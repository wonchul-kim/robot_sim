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



### camera =========================================================================
import omni.usd
from pxr import UsdPhysics

def get_robot_root_path(robot):
    # Articulation 인스턴스 / Cfg 모두 대응
    if hasattr(robot, "prim_path"):
        return robot.prim_path
    if hasattr(robot, "cfg") and hasattr(robot.cfg, "prim_path"):
        return robot.cfg.prim_path
    # 마지막 안전장치: 여러분이 지정한 경로가 있다면 하드코딩
    return "/World/Origin/Robot"

def list_rigid_links_under(root_path):
    """USD Stage를 직접 순회하며 RigidBody 링크 프림들의 (name, path) 리스트를 반환"""
    stage = omni.usd.get_context().get_stage()
    links = []
    # Stage.Traverse()로 전 범위를 순회한 뒤 root_path 아래만 필터
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if not p.startswith(root_path):
            continue
        # RigidBody API가 적용된 프림만 채택 (링크 본체)
        try:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                links.append((prim.GetName(), p))
        except Exception:
            pass
    # 그래도 못 찾았으면 이름에 'link'/'flange'가 들어가는 Xform을 예비 후보로
    if not links:
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            if not p.startswith(root_path):
                continue
            if prim.GetTypeName() == "Xform":
                n = prim.GetName().lower()
                if any(k in n for k in ("link", "flange", "wrist")):
                    links.append((prim.GetName(), p))
    return links

def find_ee_prim_path_version_agnostic(robot):
    root = get_robot_root_path(robot)
    links = list_rigid_links_under(root)
    if not links:
        raise RuntimeError(f"No rigid links found under {root}")

    # 선호 이름(정확 일치 → 부분 일치 → 최심도 링크)
    prefer_exact = ("tool0","tcp","ee_link","flange")
    prefer_sub   = ("tool","tcp","ee","flange","wrist_3","wrist3","link_6","link6")

    # 1) 정확 일치
    name2path = {n: p for (n, p) in links}
    for k in prefer_exact:
        if k in name2path:
            return name2path[k]

    # 2) 부분 일치
    for (n, p) in links:
        nl = n.lower()
        if any(k in nl for k in prefer_sub):
            return p

    # 3) 최심도(경로 세그먼트가 가장 긴 링크 = 말단일 가능성 높음)
    return max((p for (_, p) in links), key=lambda s: s.count("/"))

# 사용 예 (initialize/reset 이후):
ee_prim_path = find_ee_prim_path_version_agnostic(robot)
print("EE prim path:", ee_prim_path)



# Isaac Lab sensor API (버전에 따라 import 경로가 약간 다를 수 있습니다)
from isaaclab.sensors.camera import Camera, CameraCfg

cam_cfg = CameraCfg(
    prim_path=f"{ee_prim_path}/hand_camera",  # ← 부모-자식 관계는 경로로 결정!
    width=640,
    height=480,
    update_period=0.1,
    data_types=["rgb", "depth"],   # 필요 타입
    debug_vis=False,
    spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),

)

# # 버전에 따라 OffsetCfg 필드명이 다를 수 있어요.
# # 가장 흔한 두 가지 케이스를 순서대로 시도:
# try:
#     cam_cfg.offset = CameraCfg.OffsetCfg(translation=(0.0, 0.0, 0.10), rotation=(0.0, 0.0, 0.0))
# except TypeError:
#     cam_cfg.offset = CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.10), rot=(0.0, 0.0, 0.0))

hand_cam = Camera(cam_cfg)  # (어떤 빌드는 Camera(cfg=cam_cfg) 형태. 둘 다 안 되면 이 줄만 바꿔보세요.)
try:
    hand_cam.initialize(sim)   # _is_outdated 등 내부 버퍼 생성
except TypeError:
    hand_cam.initialize()      # 일부 빌드는 인자 없이

hand_cam.reset()

    
q0 = robot.data.default_joint_pos.clone().cpu().detach()  # shape: [1, num_joints]
print(f'Current Joint Pos: {robot.data.joint_pos}')
qd0 = torch.zeros_like(q0)

amp = torch.tensor([[0.2]*num_joints])  # 0.2rad ≈ 11.5deg
freq = 0.2  # Hz
dt = sim.get_physics_dt() if hasattr(sim, "get_physics_dt") else 1.0/60.0
t = 0.0

import numpy as np
from robot_sim.isaaclab.utils.articulations import get_ee_state_world

sim.play()
steps = 500
for i in range(steps):
    t += dt 
    q_target = q0 + amp * torch.sin(torch.tensor([[2*math.pi*freq*t]])) 
    qd_target = torch.zeros_like(q_target)

    ee = get_ee_state_world(robot, ee_name="tool0", env_idx=0, to_numpy=True)
    robot.write_joint_state_to_sim(q_target, qd_target)

    # === 카메라 프레임 꺼내기 ===
    # 버전에 따라 인터페이스가 약간 다릅니다.
    # (hand_cam.data.output["rgb"] 또는 hand_cam.get_rgba() 등)
    rgb = hand_cam.data.output["rgb"]        # Tensor [N, H, W, 4] or [H, W, 4]
    depth = hand_cam.data.output.get("depth", None)

    # 예: 첫 환경의 RGB 프레임 numpy로 꺼내기
    rgb_np = rgb[0].cpu().numpy() if rgb.ndim == 4 else rgb.cpu().numpy()




    sim.step()
    
sim.stop()

