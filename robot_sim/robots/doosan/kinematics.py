import numpy as np

# Joint sequence: joint_1 .. joint_6 (base_link -> tool0)
ACTUATED = [
    {"name":"joint_1", "xyz": [0.0, 0.0, 0.1525], "rpy": [0.0, 0.0, 0.0], "axis": [0.0,0.0,1.0]},
    {"name":"joint_2", "xyz": [0.0, 0.0345, 0.0], "rpy": [0.0,-1.571,-1.571], "axis": [0.0,0.0,1.0]},
    {"name":"joint_3", "xyz": [0.62, 0.0, 0.0], "rpy": [0.0,0.0,1.571], "axis": [0.0,0.0,1.0]},
    {"name":"joint_4", "xyz": [0.0, -0.559, 0.0], "rpy": [1.571,0.0,0.0], "axis": [0.0,0.0,1.0]},
    {"name":"joint_5", "xyz": [0.0, 0.0, 0.0], "rpy": [-1.571,0.0,0.0], "axis": [0.0,0.0,1.0]},
    {"name":"joint_6", "xyz": [0.0, -0.121, 0.0], "rpy": [1.571,0.0,0.0], "axis": [0.0,0.0,1.0]},
]

# Fixed transform from link_6 to tool0
TOOL_XYZ = [0.0,0.0,0.0]
TOOL_RPY = [np.pi, -np.pi/2, 0.0]

def rpy_to_R(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                  [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                  [-sp,   cp*sr,             cp*cr]])
    return R

def T_from_xyz_rpy(xyz, rpy):
    T = np.eye(4)
    T[:3,:3] = rpy_to_R(*rpy)
    T[:3,3] = np.array(xyz, dtype=float)
    return T

def axis_angle_T(axis, theta):
    ax = np.array(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    x, y, z = ax
    c, s, C = np.cos(theta), np.sin(theta), 1 - np.cos(theta)
    R = np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ]
    ])
    T = np.eye(4); T[:3,:3] = R
    return T

def fk(q):
    """Return 4x4 transform from base_link to tool0 for joint vector q (len=6, rad)."""
    if len(q) != 6:
        raise ValueError("Expected 6 joint angles (rad).")
    T = np.eye(4)
    for j, qi in zip(ACTUATED, q):
        T = T @ T_from_xyz_rpy(j["xyz"], j["rpy"]) @ axis_angle_T(j["axis"], qi)
    T = T @ T_from_xyz_rpy(TOOL_XYZ, TOOL_RPY)
    return T

def pose(T):
    """Return (position[m], roll-pitch-yaw[rad]) from a 4x4 transform."""
    p = T[:3,3].copy()
    r11,r12,r13 = T[0,0], T[0,1], T[0,2]
    r21,r22,r23 = T[1,0], T[1,1], T[1,2]
    r31,r32,r33 = T[2,0], T[2,1], T[2,2]
    pitch = -np.arcsin(r31)
    roll = np.arctan2(r32/np.cos(pitch), r33/np.cos(pitch))
    yaw = np.arctan2(r21/np.cos(pitch), r11/np.cos(pitch))
    return p, np.array([roll, pitch, yaw])

def fk_delta(delta_q, q0=None):
    """Apply delta joint angles (rad) to q0 (defaults zeros), return (T, p, rpy)."""
    if q0 is None:
        q0 = np.zeros(6)
    q = np.array(q0, dtype=float) + np.array(delta_q, dtype=float)
    T = fk(q)
    p, rpy = pose(T)
    return T, p, rpy


if __name__ == "__main__":
    import pandas as pd 
    import ast
    from scipy.spatial.transform import Rotation as R

    traj = pd.read_csv('/HDD/etc/outputs/isaac/trajectory.csv')
    print(traj)
    

    joint_position = traj['joint_position'].apply(ast.literal_eval)
    joint_velocity = traj['joint_velocity'].apply(ast.literal_eval)
    ee_pos_w = traj['ee_pos_w'].apply(ast.literal_eval)
    ee_quat_w_xyzw = traj['ee_quat_w_xyzw'].apply(ast.literal_eval)

    print(joint_position[0])
    print(ee_pos_w[0])

    q = joint_position[0]
    T = fk(q)
    p, rpy = pose(T)
    print("pos(m) =", p, "rpy(rad) =", rpy)


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
        