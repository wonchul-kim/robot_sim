import torch
import numpy as np

def _first_attr(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return n, getattr(obj, n)
    raise AttributeError(f"None of attrs {names} found on {obj}")

def _get_name_list(robot):
    for field in ["link_names", "body_names", "rigid_body_names", "names_links", "names_bodies"]:
        if hasattr(robot.data, field):
            return getattr(robot.data, field), field
    for field in ["link_names", "body_names", "rigid_body_names"]:
        if hasattr(robot, field):
            return getattr(robot, field), field
    raise AttributeError("No link/body name list found on robot.data or robot")

def _get_state_tensors(robot):
    data = robot.data
    pos_name,  pos  = _first_attr(data, "link_pos_w",  "body_pos_w",  "rigid_body_pos_w")
    quat_name, quat = _first_attr(data, "link_quat_w", "body_quat_w", "rigid_body_quat_w")
    lin_name,  lin  = _first_attr(data, "link_lin_vel_w","body_lin_vel_w","rigid_body_lin_vel_w")
    ang_name,  ang  = _first_attr(data, "link_ang_vel_w","body_ang_vel_w","rigid_body_ang_vel_w")
    return {"pos": pos, "quat": quat, "lin": lin, "ang": ang,
            "pos_name": pos_name, "quat_name": quat_name, "lin_name": lin_name, "ang_name": ang_name}

def _resolve_ee_index(name_list, ee_name="tool0"):
    if ee_name in name_list:
        return name_list.index(ee_name)
    for i, n in enumerate(name_list):
        last = n.split("/")[-1]
        if last == ee_name or n.endswith("/"+ee_name):
            return i
    aliases = [ee_name, ee_name+"_link", "tcp", "ee_link", "end_effector", "EE", "Tool0", "TOOL0"]
    for alias in aliases:
        for i, n in enumerate(name_list):
            last = n.split("/")[-1]
            if last == alias or n.endswith("/"+alias):
                return i
    raise ValueError(f"Could not find ee link by name '{ee_name}'. Sample names: {name_list[:10]} ... (total {len(name_list)})")

def _normalize_quat_xyzw(q):
    # q: (..., 4) assumed XYZW from Isaac
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    return q

def get_ee_state_world(robot, ee_name="tool0", env_idx=0, to_numpy=True):
    names, used_field = _get_name_list(robot)
    st = _get_state_tensors(robot)
    # sanity: 같은 계열의 이름/텐서인지(둘 다 link_* 또는 body_* 등)
    assert used_field.split("_")[0] in st["pos_name"], \
        f"name list ({used_field}) and tensors ({st['pos_name']}) may not align"

    ee_idx = _resolve_ee_index(names, ee_name)

    ee_pos  = st["pos"][env_idx, ee_idx]
    ee_quat = st["quat"][env_idx, ee_idx]
    # ee_quat = ee_quat[torch.tensor([3, 0, 1, 2])]
    ee_lin  = st["lin"][env_idx, ee_idx]
    ee_ang  = st["ang"][env_idx, ee_idx]

    ee_quat = _normalize_quat_xyzw(ee_quat)

    if to_numpy:
        return {
            "name_field": used_field,
            "tensor_fields": (st["pos_name"], st["quat_name"], st["lin_name"], st["ang_name"]),
            "index": ee_idx,
            "pos_w":  ee_pos.detach().cpu().numpy(),      # shape (3,)
            "quat_w_xyzw": ee_quat.detach().cpu().numpy(),# shape (4,)
            "lin_w":  ee_lin.detach().cpu().numpy(),
            "ang_w":  ee_ang.detach().cpu().numpy(),
        }
    else:
        return {
            "name_field": used_field,
            "tensor_fields": (st["pos_name"], st["quat_name"], st["lin_name"], st["ang_name"]),
            "index": ee_idx,
            "pos_w":  ee_pos,
            "quat_w_xyzw": ee_quat,
            "lin_w":  ee_lin,
            "ang_w":  ee_ang,
        }
