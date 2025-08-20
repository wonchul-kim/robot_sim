import numpy as np
import math
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

class DoosanM1013Kinematics:
    def __init__(self):
        ### DH Parameters (URDF 기반, 단위: m, radian)
        self.dh_params = np.array([
            [0,      np.pi/2,   0.159,  0],         # Joint 1 (Base) - 159mm -> 0.159m
            [0,      0,         0,      -np.pi/2],  # Joint 2 (Shoulder)
            [0.409,  0,         0.154,  0],         # Joint 3 (Elbow) - 409mm -> 0.409m, 154mm -> 0.154m
            [0.367,  np.pi/2,   0,      0],         # Joint 4 (Wrist 1) - 367mm -> 0.367m
            [0,      -np.pi/2,  0.122,  0],         # Joint 5 (Wrist 2) - 122mm -> 0.122m
            [0,      0,         0.106,  0]          # Joint 6 (Wrist 3) - 106mm -> 0.106m
        ])

        # Joint limits (radian)
        self.joint_limits = np.array([
            [-2*np.pi, 2*np.pi],                    # J1: ±360°
            [-2*np.pi, 2*np.pi],                    # J2: ±360° 
            [-160*np.pi/180, 160*np.pi/180],        # J3: ±160°
            [-2*np.pi, 2*np.pi],                    # J4: ±360°
            [-2*np.pi, 2*np.pi],                    # J5: ±360°
            [-2*np.pi, 2*np.pi]                     # J6: ±360°
        ])

        # 로봇 스펙 (단위: m, kg)
        self.specs = {
            'model': 'M1013',
            'payload': 10,        # kg
            'reach': 1.3,         # m (1300mm -> 1.3m)
            'repeatability': 5e-5,  # m (0.05mm -> 5e-5m)
            'weight': 33,         # kg
            'axes': 6
        }

        # IK 설정
        self.ik_tolerance = 1e-9  # m (1e-6mm -> 1e-9m)
        self.ik_max_iterations = 100

    def dh_transform(self, a, alpha, d, theta):
        """DH 파라미터를 이용한 동차 변환 행렬 계산"""
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)

        T = np.array([
            [cos_theta, -sin_theta*cos_alpha,  sin_theta*sin_alpha, a*cos_theta],
            [sin_theta,  cos_theta*cos_alpha, -cos_theta*sin_alpha, a*sin_theta],
            [0,          sin_alpha,            cos_alpha,           d],
            [0,          0,                    0,                   1]
        ])

        return T

    def forward_kinematics(self, joint_angles, unit='rad'):
        """
        Forward Kinematics 계산

        Args:
            joint_angles: 6개 조인트 각도 리스트
            unit: 'rad' 또는 'deg'

        Returns:
            dict: end-effector 위치/자세 정보 (위치는 m 단위)
        """
        if len(joint_angles) != 6:
            raise ValueError("6개의 조인트 각도가 필요합니다")

        # 각도 단위 변환
        if unit == 'deg':
            angles = np.array(joint_angles) * np.pi / 180
        else:
            angles = np.array(joint_angles)

        # 각 조인트의 변환 행렬 계산
        transforms = []
        T_total = np.eye(4)

        for i in range(6):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = angles[i] + theta_offset

            T_i = self.dh_transform(a, alpha, d, theta)
            transforms.append(T_i)
            T_total = np.dot(T_total, T_i)

        # End-effector 위치와 방향 추출
        position = T_total[:3, 3]  # 미터 단위
        rotation_matrix = T_total[:3, :3]

        # 오일러 각도 계산 (ZYX convention)
        sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0

        euler_angles = np.array([x, y, z])

        result = {
            'robot_model': self.specs['model'],
            'input_joints': joint_angles,
            'input_unit': unit,
            'joint_angles_rad': angles,
            'individual_transforms': transforms,
            'final_transform': T_total,
            'position': {
                'x': position[0],     # m
                'y': position[1],     # m
                'z': position[2],     # m
                'xyz': position       # m
            },
            'orientation': {
                'rotation_matrix': rotation_matrix,
                'euler_xyz_rad': euler_angles,
                'euler_xyz_deg': euler_angles * 180 / np.pi
            },
            'reach_distance': np.linalg.norm(position[:2]),  # m
            'total_distance': np.linalg.norm(position)       # m
        }

        return result

    def compute_jacobian(self, joint_angles, unit='rad'):
        """Jacobian 행렬 계산 (수치적 미분)"""
        delta = 1e-6

        current_result = self.forward_kinematics(joint_angles, unit=unit)
        current_pos = current_result['position']['xyz']
        current_rot = current_result['orientation']['euler_xyz_rad']
        current_pose = np.concatenate([current_pos, current_rot])

        jacobian = np.zeros((6, 6))

        for i in range(6):
            joints_plus = list(joint_angles)
            if unit == 'deg':
                joints_plus[i] += delta * 180 / np.pi
            else:
                joints_plus[i] += delta

            result_plus = self.forward_kinematics(joints_plus, unit=unit)
            pos_plus = result_plus['position']['xyz'] 
            rot_plus = result_plus['orientation']['euler_xyz_rad']
            pose_plus = np.concatenate([pos_plus, rot_plus])

            if unit == 'deg':
                jacobian[:, i] = (pose_plus - current_pose) / (delta * 180 / np.pi)
            else:
                jacobian[:, i] = (pose_plus - current_pose) / delta

        return jacobian

    def inverse_kinematics(self, target_position, target_orientation=None, 
                          initial_guess=None, unit='rad', method='jacobian'):
        """
        Inverse Kinematics 계산

        Args:
            target_position: 목표 위치 [x, y, z] (m)
            target_orientation: 목표 자세 [rx, ry, rz] (rad) - None이면 위치만 제어
            initial_guess: 초기 추정값
            unit: 'rad' 또는 'deg'
            method: 'jacobian' 또는 'numerical'

        Returns:
            dict: IK 결과 (위치는 m 단위)
        """
        if method == 'jacobian':
            return self._inverse_kinematics_jacobian(
                target_position, target_orientation, initial_guess, unit
            )
        else:
            return self._inverse_kinematics_numerical(
                target_position, target_orientation, initial_guess, unit
            )

    def _inverse_kinematics_jacobian(self, target_position, target_orientation=None,
                                   initial_guess=None, unit='rad'):
        """Jacobian 기반 Newton-Raphson IK"""
        if initial_guess is None:
            current_joints = [0, 0, 0, 0, 0, 0]
        else:
            current_joints = list(initial_guess)

        # 목표 pose 설정
        if target_orientation is None:
            target_pose = np.array(target_position)
            position_only = True
        else:
            target_pose = np.concatenate([target_position, target_orientation])
            position_only = False

        for iteration in range(self.ik_max_iterations):
            # 현재 pose 계산
            current_result = self.forward_kinematics(current_joints, unit=unit)
            current_pos = current_result['position']['xyz']

            if position_only:
                current_pose = current_pos
                pose_error = target_pose - current_pose
                jacobian = self.compute_jacobian(current_joints, unit=unit)[:3, :]
            else:
                current_rot = current_result['orientation']['euler_xyz_rad']
                current_pose = np.concatenate([current_pos, current_rot])
                pose_error = target_pose - current_pose
                jacobian = self.compute_jacobian(current_joints, unit=unit)

            # 수렴 확인
            error_norm = np.linalg.norm(pose_error)
            if error_norm < self.ik_tolerance:
                break

            # Jacobian 역행렬 계산
            try:
                jacobian_pinv = np.linalg.pinv(jacobian)
                delta_joints = jacobian_pinv @ pose_error

                # 스텝 크기 제한
                step_limit = 0.1 if unit == 'rad' else 5.0
                delta_joints = np.clip(delta_joints, -step_limit, step_limit)

                # 조인트 각도 업데이트
                for i in range(6):
                    current_joints[i] += delta_joints[i]

            except np.linalg.LinAlgError:
                break

        # 최종 결과
        final_result = self.forward_kinematics(current_joints, unit=unit)
        final_pos = final_result['position']['xyz']
        position_error = np.linalg.norm(final_pos - target_position)

        return {
            'success': error_norm < self.ik_tolerance,
            'joint_angles': current_joints,
            'unit': unit,
            'target_position': target_position,  # m
            'target_orientation': target_orientation,
            'achieved_position': final_pos.tolist(),  # m
            'achieved_orientation': final_result['orientation']['euler_xyz_rad'].tolist() if not position_only else None,
            'position_error': position_error,  # m
            'total_error': error_norm,
            'iterations': iteration + 1,
            'position_only': position_only,
            'method': 'Jacobian Newton-Raphson'
        }

    def _inverse_kinematics_numerical(self, target_position, target_orientation=None,
                                    initial_guess=None, unit='rad'):
        """수치적 최적화 기반 IK"""
        if initial_guess is None:
            initial_guess = [0, 0, 0, 0, 0, 0]

        # 목표 pose 설정
        if target_orientation is None:
            target_pose = np.array(target_position)
            position_only = True
        else:
            target_pose = np.concatenate([target_position, target_orientation])
            position_only = False

        def cost_function(joint_angles):
            try:
                result = self.forward_kinematics(joint_angles, unit=unit)
                current_pos = result['position']['xyz']

                if position_only:
                    error = current_pos - target_pose
                else:
                    current_rot = result['orientation']['euler_xyz_rad']
                    current_pose = np.concatenate([current_pos, current_rot])
                    error = current_pose - target_pose

                return error
            except:
                return np.full_like(target_pose, 1e6)

        try:
            solution, info, ier, msg = fsolve(cost_function, initial_guess, full_output=True)

            final_error = cost_function(solution)
            error_norm = np.linalg.norm(final_error)

            verification = self.forward_kinematics(solution, unit=unit)

            return {
                'success': ier == 1 and error_norm < self.ik_tolerance,
                'joint_angles': solution.tolist(),
                'unit': unit,
                'target_position': target_position,  # m
                'target_orientation': target_orientation,
                'achieved_position': verification['position']['xyz'].tolist(),  # m
                'achieved_orientation': verification['orientation']['euler_xyz_rad'].tolist() if not position_only else None,
                'position_error': np.linalg.norm(verification['position']['xyz'] - target_position),  # m
                'total_error': error_norm,
                'iterations': info['nfev'],
                'solver_message': msg,
                'position_only': position_only,
                'method': 'Numerical Optimization'
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'joint_angles': None,
                'unit': unit,
                'method': 'Numerical Optimization'
            }

    def move_delta_joints(self, current_joints, delta_joints, unit='deg'):
        """델타 조인트 움직임 (강화학습용)"""
        if len(current_joints) != 6 or len(delta_joints) != 6:
            raise ValueError("6개의 조인트 값이 필요합니다")

        new_joints = [current + delta for current, delta in zip(current_joints, delta_joints)]
        result = self.forward_kinematics(new_joints, unit=unit)

        result['delta_movement'] = {
            'previous_joints': current_joints,
            'delta_joints': delta_joints,
            'new_joints': new_joints
        }

        return result

    def move_to_position(self, target_position, current_joints=None, unit='deg'):
        """목표 위치로 직접 이동 (IK 사용)

        Args:
            target_position: 목표 위치 [x, y, z] (m)
            current_joints: 현재 조인트 각도
            unit: 'rad' 또는 'deg'
        """
        initial_guess = current_joints if current_joints is not None else [0, 0, 0, 0, 0, 0]

        ik_result = self.inverse_kinematics(
            target_position=target_position,
            initial_guess=initial_guess,
            unit=unit,
            method='jacobian'
        )

        return ik_result

    def print_specs(self):
        """로봇 스펙 정보 출력"""
        print(f"=== {self.specs['model']} Robot Specifications ===")
        print(f"model: {self.specs['model']}")
        print(f"payload: {self.specs['payload']} kg")
        print(f"reach: {self.specs['reach']} m")
        print(f"repeatability: {self.specs['repeatability']*1000:.3f} mm")
        print(f"weight: {self.specs['weight']} kg")
        print(f"axes: {self.specs['axes']}")

        print("\nJoint Limits:")
        for i, limits in enumerate(self.joint_limits):
            print(f"  J{i+1}: [{limits[0]*180/np.pi:6.1f}°, {limits[1]*180/np.pi:6.1f}°]")

        print("\nDH Parameters (URDF based, lengths in meters):")
        print("  Joint |    a    |  alpha  |    d    | offset")
        print("  ------|---------|---------|---------|--------")
        for i, params in enumerate(self.dh_params):
            print(f"    J{i+1}  | {params[0]:7.3f} | {params[1]:7.3f} | {params[2]:7.3f} | {params[3]:6.3f}")

# 강화학습 환경 클래스
class M1013ReinforcementLearningEnvironment:
    def __init__(self):
        """M1013 로봇 강화학습 환경 (단위: m)"""
        self.robot = DoosanM1013Kinematics()
        self.current_joints = [0, 0, 0, 0, 0, 0]  # 초기 위치
        self.target_position = None

    def reset(self, initial_joints=None):
        """환경 초기화"""
        if initial_joints is None:
            self.current_joints = [0, 0, 0, 0, 0, 0]
        else:
            self.current_joints = list(initial_joints)

        return self.get_observation()

    def get_observation(self):
        """현재 상태 관측 (위치는 m 단위)"""
        fk_result = self.robot.forward_kinematics(self.current_joints, unit='deg')
        return {
            'joint_angles': self.current_joints,
            'end_effector_position': fk_result['position']['xyz'].tolist(),  # m
            'end_effector_orientation': fk_result['orientation']['euler_xyz_deg'].tolist()
        }

    def step_delta_joints(self, delta_joints):
        """델타 조인트 액션 실행"""
        result = self.robot.move_delta_joints(
            self.current_joints, 
            delta_joints, 
            unit='deg'
        )

        self.current_joints = result['delta_movement']['new_joints']

        return {
            'observation': self.get_observation(),
            'info': result
        }

    def step_target_position(self, target_position):
        """목표 위치로 직접 이동 (IK 사용)

        Args:
            target_position: 목표 위치 [x, y, z] (m)
        """
        ik_result = self.robot.move_to_position(
            target_position, 
            self.current_joints, 
            unit='deg'
        )

        if ik_result['success']:
            self.current_joints = ik_result['joint_angles']

        return {
            'success': ik_result['success'],
            'observation': self.get_observation(),
            'ik_result': ik_result
        }

    def set_target(self, target_position):
        """목표 위치 설정 (m)"""
        self.target_position = target_position

    def get_reward(self):
        """보상 계산 (목표 위치와의 거리 기반)"""
        if self.target_position is None:
            return 0

        current_pos = self.get_observation()['end_effector_position']
        distance = np.linalg.norm(np.array(current_pos) - np.array(self.target_position))  # m

        # 거리 기반 보상 (가까울수록 높은 보상)
        reward = -distance  # m 단위

        # 목표 도달시 보너스
        if distance < 0.005:  # 5mm = 0.005m 이내
            reward += 10.0

        return reward

# 사용 예제
def example_usage():
    """사용 예제 (모든 길이는 m 단위)"""
    print("=== Doosan M1013 Complete Kinematics Example (meters) ===")

    # 로봇 인스턴스 생성
    robot = DoosanM1013Kinematics()
    robot.print_specs()

    # Forward Kinematics
    print("\n1. Forward Kinematics:")
    joints = [30, -20, 45, 0, 25, 60]
    fk_result = robot.forward_kinematics(joints, unit='deg')
    print(f"   Joints: {joints} deg")
    print(f"   Position: [{fk_result['position']['x']:.4f}, {fk_result['position']['y']:.4f}, {fk_result['position']['z']:.4f}] m")
    print(f"   Reach: {fk_result['reach_distance']:.4f} m")

    # Inverse Kinematics
    print("\n2. Inverse Kinematics:")
    target = [-0.2, -0.3, -0.6]  # m
    ik_result = robot.inverse_kinematics(target, unit='deg')
    print(f"   Target: {target} m")
    print(f"   Success: {ik_result['success']}")
    if ik_result['success']:
        print(f"   Joints: {[round(x, 1) for x in ik_result['joint_angles']]} deg")
        print(f"   Error: {ik_result['position_error']:.6f} m")
        print(f"   Achieved: {[round(x, 4) for x in ik_result['achieved_position']]} m")
    fk_res = robot.forward_kinematics([round(x, 1) for x in ik_result['joint_angles']], unit='deg')
    print("Evaluation: ", np.allclose(fk_res['position']['xyz'], target, atol=1e5))
    
    # 강화학습 환경
    print("\n3. Reinforcement Learning Environment:")
    rl_env = M1013ReinforcementLearningEnvironment()
    obs = rl_env.reset()
    print(f"   Initial position: {[round(x, 4) for x in obs['end_effector_position']]} m")

    # 델타 움직임
    delta_action = [5, -3, 10, 0, 2, 0]  # deg
    step_result = rl_env.step_delta_joints(delta_action)
    new_pos = step_result['observation']['end_effector_position']
    print(f"   After delta {delta_action} deg: {[round(x, 4) for x in new_pos]} m")

    # 목표 위치 이동
    target_pos = [-0.15, -0.25, -0.5]  # m
    target_result = rl_env.step_target_position(target_pos)
    if target_result['success']:
        final_pos = target_result['observation']['end_effector_position']
        print(f"   Target move to {target_pos} m: {[round(x, 4) for x in final_pos]} m")
        print(f"   Error: {target_result['ik_result']['position_error']:.6f} m")

if __name__ == "__main__":
    example_usage()
