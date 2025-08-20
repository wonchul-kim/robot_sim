import torch

def _first_attr(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return n, getattr(obj, n)
    raise AttributeError(f"None of attrs {names} found on {obj}")

def _get_name_list(robot):
    # 가장 흔한 이름 순으로 검색
    for field in ["link_names", "body_names", "rigid_body_names", "names_links", "names_bodies"]:
        if hasattr(robot.data, field):
            return getattr(robot.data, field), field
    # Articulation 자체에 있을 수도 있음
    for field in ["link_names", "body_names", "rigid_body_names"]:
        if hasattr(robot, field):
            return getattr(robot, field), field
    raise AttributeError("No link/body name list found on robot.data or robot")

def _get_state_tensors(robot):
    data = robot.data
    # 위치/자세/속도 텐서 이름 자동탐색 (world frame)
    pos_name,  pos  = _first_attr(data, "link_pos_w",  "body_pos_w",  "rigid_body_pos_w")
    quat_name, quat = _first_attr(data, "link_quat_w", "body_quat_w", "rigid_body_quat_w")
    lin_name,  lin  = _first_attr(data, "link_lin_vel_w","body_lin_vel_w","rigid_body_lin_vel_w")
    ang_name,  ang  = _first_attr(data, "link_ang_vel_w","body_ang_vel_w","rigid_body_ang_vel_w")
    return {"pos": pos, "quat": quat, "lin": lin, "ang": ang,
            "pos_name": pos_name, "quat_name": quat_name, "lin_name": lin_name, "ang_name": ang_name}

def _resolve_ee_index(name_list, ee_name="tool0"):
    # 정확 일치 우선
    if ee_name in name_list:
        return name_list.index(ee_name)
    # 끝부분 일치 (USD 경로 포함되는 경우)
    for i, n in enumerate(name_list):
        if n.endswith("/"+ee_name) or n.split("/")[-1] == ee_name:
            return i
    # 흔한 변형들 시도
    aliases = [ee_name, ee_name+"_link", "tcp", "ee_link", "end_effector", "EE", "Tool0", "TOOL0"]
    for alias in aliases:
        for i, n in enumerate(name_list):
            if n.endswith("/"+alias) or n.split("/")[-1] == alias:
                return i
    raise ValueError(f"Could not find ee link by name '{ee_name}'. Available names include e.g.: {name_list[:10]} ... (total {len(name_list)})")

def get_ee_state_world(robot, ee_name="tool0", env_idx=0, to_numpy=True):
    names, used_field = _get_name_list(robot)
    st = _get_state_tensors(robot)
    ee_idx = _resolve_ee_index(names, ee_name)
    ee_pos  = st["pos"][env_idx, ee_idx]
    ee_quat = st["quat"][env_idx, ee_idx]
    ee_lin  = st["lin"][env_idx, ee_idx]
    ee_ang  = st["ang"][env_idx, ee_idx]
    if to_numpy:
        return {
            "name_field": used_field,
            "tensor_fields": (st["pos_name"], st["quat_name"], st["lin_name"], st["ang_name"]),
            "index": ee_idx,
            "pos_w":  ee_pos.detach().cpu().numpy(),
            "quat_w": ee_quat.detach().cpu().numpy(),  # (x,y,z,w)일 가능성 높음
            "lin_w":  ee_lin.detach().cpu().numpy(),
            "ang_w":  ee_ang.detach().cpu().numpy(),
        }
    else:
        return {
            "name_field": used_field,
            "tensor_fields": (st["pos_name"], st["quat_name"], st["lin_name"], st["ang_name"]),
            "index": ee_idx,
            "pos_w":  ee_pos,
            "quat_w": ee_quat,
            "lin_w":  ee_lin,
            "ang_w":  ee_ang,
        }

