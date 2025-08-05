import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time 

#     def set_joint_positions(self, positions):
#         if len(positions) != len(self.joint_names):
#             self.get_logger().error(f"Positions length mismatch.")
#             return
#         self.current_positions = positions
#         self.get_logger().info(f"Updated joint positions: {positions}")

# def main(args=None):
#     rclpy.init(args=args)
#     node = JointPublisher()

#     try:
#         while rclpy.ok():
#             try:
#                 import time 
#                 time.sleep(1)
#                 parts = [x + 0.01 for x in node.current_positions]
#                 positions = [float(p) for p in parts]
#             except ValueError:
#                 print("Invalid input. Enter numeric values.")
#                 continue
#             node.set_joint_positions(positions)
#     except KeyboardInterrupt:
#         pass

#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()


# class JointPublisher(Node):
#     def __init__(self):
#         super().__init__('joint_publisher')
#         self.pub = self.create_publisher(JointState, '/joint_command', 10)
#         self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 
#                             'elbow_joint', 
#                             'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
#         self.current_positions = [0.0] * len(self.joint_names)
#         self.timer_period = 0.1  # 10Hz
#         self.timer = self.create_timer(self.timer_period, self.timer_callback)
#         self.start_time = time.time()


#     def timer_callback(self):
#         msg = JointState()
#         msg.header.stamp = self.get_clock().now().to_msg()
#         msg.name = self.joint_names
#         # msg.position = self.current_positions
#         msg.position = [0.01, 0.01, 0.01, 0.02, 0.02, 0.02]
#         self.pub.publish(msg)
#         self.get_logger().info(f'Published joint positions: {msg.position}')

# def main(args=None):
#     rclpy.init(args=args)
#     publisher = JointPublisher()
#     rclpy.spin(publisher)
    
#     while True:
#         print((time.time() - publisher.start_time)%5)
#         if (time.time() - publisher.start_time)%5 < 1:
#             publisher.current_positions = [x + 0.01 for x in publisher.current_positions]
    
#         if time.time() - publisher.start_time > 60:
#             break
    
#     publisher.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

import time

class JointPublisher(Node):
    def __init__(self):
        super().__init__('joint_publisher')
        self.publisher_ = self.create_publisher(JointState, '/joint_command', 10)
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint',
            'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

    def publish_joint_positions(self, positions):
        if len(positions) != len(self.joint_names):
            self.get_logger().error("positions 길이가 joint 개수와 다릅니다.")
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = positions
        self.publisher_.publish(msg)
        # self.get_logger().info(f'Published joint positions: {positions}')

def main(args=None):
    rclpy.init(args=args)
    publisher = JointPublisher()

    # 원하는 joint 값으로 퍼블리시(함수 호출)
    publisher.publish_joint_positions([0.1, 0.2, 0.3, 0.1, 0.0, 0.0])
    time.sleep(3)
    publisher.publish_joint_positions([0.5, 0.0, -0.2, 0.2, 0.1, 0.0])

    # rclpy.spin()을 이용해 ROS 콜백 루프 유지 (서비스/구독 등 쓰지 않으면 짧게 대기 후 종료 가능)
    time.sleep(1)  # 퍼블리시 후 잠시 대기

    publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
