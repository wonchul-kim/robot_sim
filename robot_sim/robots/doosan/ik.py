import numpy as np
from fk import ACTUATED, T_from_xyz_rpy, axis_angle_T, TOOL_RPY, TOOL_XYZ, fk, fk_delta, pose

# ====== IK 유틸 ======

def chain_forward_all(q):
    """
    각 관절 적용 '전'의 고정 변환 * 회전(조인트) '전까지'를 누적해
    - 각 관절 원점 o_i (base frame)
    - 각 관절 회전축 z_i (base frame)
    - 말단(툴) 포즈 T_ee
    를 반환.
    """
    T = np.eye(4)
    origins = []
    z_axes = []

    for j, qi in zip(ACTUATED, q):
        # 고정 변환으로 이동 (관절 i의 원점, z축은 이 고정변환 후의 Z가 아님! 조인트 회전축은 axis를 회전행렬로 보낼 것)
        T = T @ T_from_xyz_rpy(j["xyz"], j["rpy"])
        # 이 시점의 원점
        origins.append(T[:3,3].copy())
        # 이 시점에서 관절축(로컬 axis)을 베이스 프레임으로
        z_axes.append((T[:3,:3] @ np.array(j["axis"], float)).copy())
        # 관절 회전 적용
        T = T @ axis_angle_T(j["axis"], qi)

    # 툴(고정) 변환
    T = T @ T_from_xyz_rpy(TOOL_XYZ, TOOL_RPY)
    return np.array(origins).T, np.array(z_axes).T, T

def skew(v):
    x,y,z = v
    return np.array([[0, -z, y],
                     [z,  0,-x],
                     [-y, x, 0]])

def rot_log(R):
    """
    SO(3) 로그맵: 회전행렬 -> so(3) 벡터 (axis * angle)
    수치적으로 안전한 근사 포함.
    """
    cos_theta = (np.trace(R) - 1) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3)
    w_hat = (R - R.T) / (2*np.sin(theta))
    # vee
    return theta * np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]])

def jacobian(q):
    origins, z_axes, T_ee = chain_forward_all(q)
    o_n = T_ee[:3,3]
    # origins, z_axes: shape (3,N). 아래는 (N,3)
    Jv = np.cross(z_axes.T, (o_n - origins.T))  # (N,3)
    Jw = z_axes.T                                # (N,3)
    # => 6×N으로 반환 (열이 각 관절)
    J = np.vstack([Jv.T, Jw.T])                  # (6, N)
    return J, T_ee


def ik_solve(
    T_target=None,
    p_target=None,
    R_target=None,
    q0=None,
    pos_w=1.0,
    rot_w=1.0,
    damping=1e-3,
    max_iters=200,
    tol_pos=1e-4,
    tol_rot=1e-4
):
    """
    DLS IK.
    - 입력: T_target(4x4) 또는 p_target(3,), R_target(3x3)
    - 출력: q, success, info(dict)
    """
    if T_target is None:
        if p_target is None or R_target is None:
            raise ValueError("Provide T_target or both p_target and R_target.")
        T_target = np.eye(4)
        T_target[:3,:3] = R_target
        T_target[:3,3]  = p_target
    else:
        p_target = T_target[:3,3]
        R_target = T_target[:3,:3]

    if q0 is None:
        q = np.zeros(len(ACTUATED))
    else:
        q = np.array(q0, float).copy()

    lam2 = damping**2
    for it in range(max_iters):
        J, T_now = jacobian(q)             # J: (6,N)
        p_now = T_now[:3,3]
        R_now = T_now[:3,:3]

        e_p = p_target - p_now
        R_err = R_target @ R_now.T
        e_r = rot_log(R_err)
        e = np.hstack([e_p, e_r])

        if np.linalg.norm(e_p) < tol_pos and np.linalg.norm(e_r) < tol_rot:
            return q, True, {"iters": it, "pos_err": np.linalg.norm(e_p), "rot_err": np.linalg.norm(e_r)}

        # (선택) 태스크 가중치
        W = np.diag([pos_w, pos_w, pos_w, rot_w, rot_w, rot_w])
        e_w = W @ e
        J_w = W @ J                          # (6,N)

        A = J_w @ J.T + lam2 * np.eye(6)     # (6,6)
        dq = J.T @ np.linalg.solve(A, e_w)   # (N,)

        # (선택) 스텝 제한
        # if np.linalg.norm(dq) > 0.2:
        #     dq *= 0.2 / np.linalg.norm(dq)

        q = q + dq

    # 실패 리턴
    # 마지막 에러 첨부
    J, T_now = jacobian(q)
    e_p = p_target - T_now[:3,3]
    e_r = rot_log(R_target @ T_now[:3,:3].T)
    return q, False, {"iters": max_iters, "pos_err": np.linalg.norm(e_p), "rot_err": np.linalg.norm(e_r)}

# ====== 보조: quaternion(x,y,z,w) -> R ======
def quat_xyzw_to_R(q):
    x,y,z,w = q
    # 단위화 안전
    n = np.linalg.norm([w,x,y,z])
    if n == 0: return np.eye(3)
    w,x,y,z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w),   1-2*(x*x+y*y)],
    ])

if __name__ == "__main__":

    import pandas as pd 
    import ast
    from scipy.spatial.transform import Rotation as R

    traj = pd.read_csv('/HDD/etc/outputs/isaac/trajectory.csv')

    joint_position = traj['joint_position'].apply(ast.literal_eval)
    joint_velocity = traj['joint_velocity'].apply(ast.literal_eval)
    ee_pos_w = traj['ee_pos_w'].apply(ast.literal_eval)
    ee_quat_w_xyzw = traj['ee_quat_w_xyzw'].apply(ast.literal_eval)

    print('joint_position[0]: ', joint_position[0])
    print('ee_pos_w[0]: ', ee_pos_w[0])

    q0 = np.zeros(6)
    q_goal = np.array([0.2, -0.7, 0.4, 0.3, -0.1, 0.15])
    T_goal = fk(q_goal)
    p_goal, rpy = pose(T_goal)

    q_sol, ok, info = ik_solve(T_target=T_goal, q0=q0,
                               pos_w=1.0, rot_w=1.0, damping=1e-3,
                               max_iters=200, tol_pos=1e-5, tol_rot=1e-5)
    T_sol = fk(q_sol)
    p_sol, rpy_sol = pose(T_sol)

    print("IK success:", ok, info)
    print("pos err:", np.linalg.norm(T_goal[:3,3] - T_sol[:3,3]))
    print("rot err:", np.linalg.norm(rot_log(T_goal[:3,:3] @ T_sol[:3,:3].T)))
    print("loc err:", np.linalg.norm(p_goal - p_sol))
    print(p_goal)
    print(p_sol)

    for idx in range(1, len(joint_position)):
        
        curr_joint_pos = joint_position[idx - 1]
        next_joint_pos = joint_position[idx]
        
        curr_ee_pos_w = ee_pos_w[idx - 1]
        next_ee_pos_w = ee_pos_w[idx]
        
        curr_ee_quat_w_xyzw = ee_quat_w_xyzw[idx - 1]
        next_ee_quat_w_xyzw = ee_quat_w_xyzw[idx]
        
        
        T, p, rpy = fk_delta([a - b for a, b in zip(next_joint_pos, curr_joint_pos)], curr_joint_pos)
        print("ee position eval: ", np.allclose(p, next_ee_pos_w, atol=1e-2))
        print("rpy position eval: ", np.allclose(rpy, R.from_quat(next_ee_quat_w_xyzw).as_euler('xyz', degrees=False)))
        
        q_sol, ok, info = ik_solve(T_target=)