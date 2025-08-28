import argparse
import torch
import math


### 1. Launch Omniverse =====================================================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

### 2. Scene (ground + light + robot) =====================================================================
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
# sim_utils.DomeLightCfg(intensity=1000.0).func("/World/Light", sim_utils.DomeLightCfg())

sim_utils.DomeLightCfg(
    intensity=30000.0,   
    texture_file=None,   
    exposure=0.0
).func("/World/EnvLight", sim_utils.DomeLightCfg())

# origin xform
prim_utils.create_prim("/World/Origin", "Xform", translation=(0.0, 0.0, 0.0))

### 3. Define robot =====================================================================
from robot_sim.isaaclab.assets.robots.doosan.m1013_2f140_cfg import M1013_2F140_CFG

cfg = M1013_2F140_CFG.copy()
cfg.prim_path = "/World/Origin/Robot"

from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation

robot = Articulation(cfg=cfg)


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
    # offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.03), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    offset=CameraCfg.OffsetCfg(pos=(0.3, 0.0, 0.), rot=(1.0, 0., 0., 0), convention="ros"),

)

# # 버전에 따라 OffsetCfg 필드명이 다를 수 있어요.
# # 가장 흔한 두 가지 케이스를 순서대로 시도:
# try:
#     cam_cfg.offset = CameraCfg.OffsetCfg(translation=(0.0, 0.0, 0.10), rotation=(0.0, 0.0, 0.0))
# except TypeError:
#     cam_cfg.offset = CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.10), rot=(0.0, 0.0, 0.0))

hand_cam = Camera(cam_cfg)  # (어떤 빌드는 Camera(cfg=cam_cfg) 형태. 둘 다 안 되면 이 줄만 바꿔보세요.)

world_cam_cfg = CameraCfg(
    prim_path="/World/DebugCam",         # 로봇에 부착하지 않음(월드 프레임)
    update_period=1.0/30.0,
    height=480, width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0,
        horizontal_aperture=20.955, clipping_range=(0.03, 50.0),
    ),
    # 로봇을 내려다보는 시점으로 예시 배치 (원하시면 위치/자세 조정)
    offset=CameraCfg.OffsetCfg(
        pos=(1.8, 1.8, 1.4),              # 월드 기준 위치
        rot=(0.35, -0.61, 0.35, -0.61),   # xyzw, 대각선에서 원점 보는 예시
        convention="ros",
    ),
)
world_cam = Camera(cfg=world_cam_cfg)



# 4. Simulation =====================================================================
sim = SimulationContext(sim_utils.SimulationCfg())
sim.reset()
hand_cam.reset()
world_cam.reset()
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
from pxr import UsdPhysics
stage = omni.usd.get_context().get_stage()
for prim in stage.Traverse():
    if prim.IsA(omni.usd.UsdGeom.Xform):  # 또는 UsdGeom.Mesh
        print(prim.GetPath().pathString, prim.HasAPI(UsdPhysics.RigidBodyAPI))

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

# GUI에서만: 두 개의 뷰포트를 만들고 카메라 프림을 지정
import omni.kit.viewport.utility as vp

vp_1 = vp.create_viewport_window("World View",  width=640, height=480, x=0,   y=0)
vp_2 = vp.create_viewport_window("Hand-eye View", width=640, height=480, x=660, y=0)

# 각 뷰포트의 카메라를 센서 카메라 프림으로 바인딩
# 월드 고정 카메라(예: /World/DebugCam)
api1 = vp_1.viewport_api
api1.set_active_camera("/World/DebugCam")

# EE에 붙인 핸드아이 (예: f"{ee_prim_path}/hand_camera")
api2 = vp_2.viewport_api
api2.set_active_camera(f"{ee_prim_path}/hand_camera")


sim.play()
steps = 500  # 약 5초 @60Hz
for i in range(steps): ###  =====================================================================
    t += dt 
    # 목표 각도(부드럽게 왔다갔다)
    q_target = q0 + amp * torch.sin(torch.tensor([[2*math.pi*freq*t]]))  # [1, dof]
    
    if i < 200:
        q_target[0][-2:] = torch.tensor([0, 0], dtype=torch.float64)
    elif i <= 200 and i < 400:
        q_target[0][-2:] = torch.tensor([0.01, 0.01], dtype=torch.float64)
    elif i <= 400 and i < 600:
        q_target[0][-2:] = torch.tensor([0.015, 0.015], dtype=torch.float64)
    elif i <= 600 and i < 800:
        q_target[0][-2:] = torch.tensor([0.02, 0.02], dtype=torch.float64)
    else:
        q_target[0][-2:] = torch.tensor([0.025, 0.025], dtype=torch.float64)
        
    
    qd_target = torch.zeros_like(q_target)

    # ee = get_ee_state_world(robot, ee_name="tool0", env_idx=0, to_numpy=True)

    if i%200 == 0:
        print(f'q_target: {q_target}')
        print(f'robot.data.joint_pos: {robot.data.joint_pos}')
        print(f'robot.data.joint_vel: {robot.data.joint_vel}')
        
    #     # # 사용 예:
    #     # print("[EE] index:", ee["index"], "name_field:", ee["name_field"], "tensors:", ee["tensor_fields"])
    #     print("pos_w :", ee["pos_w"])
    #     print("quat_w:", ee["quat_w_xyzw"])
    #     print("lin_w :", ee["lin_w"])
    #     print("ang_w :", ee["ang_w"])

    # # “움직이는지 확인” 용도로 매 스텝 joint state를 조금씩 덮어쓰기
    robot.write_joint_state_to_sim(q_target, qd_target)
    
    import cv2
    import imageio.v3 as iio

    world_cam.update(dt)
    hand_cam.update(dt)

    # if i % 60 == 0:
    #     w_rgb = world_cam.data.output["rgb"][0].cpu().numpy()   # (H, W, 3) uint8
    #     h_rgb = hand_cam.data.output["rgb"][0].cpu().numpy()

    #     # 해상도가 다르면 하나를 다른 쪽 높이에 맞춰 리사이즈
    #     if w_rgb.shape[0] != h_rgb.shape[0]:
    #         scale = w_rgb.shape[0] / h_rgb.shape[0]
    #         new_w = int(h_rgb.shape[1] * scale)
    #         # OpenCV가 있으면 더 쉽지만, 의존 없애려면 numpy로 간단 리사이즈 대신 동일 해상도로 센서 설정을 맞추세요.
    #         import cv2
    #         h_rgb = cv2.resize(h_rgb, (new_w, w_rgb.shape[0]))

    #     combo = np.concatenate([w_rgb, h_rgb], axis=1)  # 가로로 붙이기
    #     iio.imwrite(f"/HDD/etc/outputs/camera/combo_{i:04d}.png", combo)

    # 한 스텝 진행
    sim.step()
    

# 종료 정리
sim.stop()
