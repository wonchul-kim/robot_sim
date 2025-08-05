import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener


class JointSubscriber(Node):
    def __init__(self):
        super().__init__('isaac_joint_subscriber')

        self.sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.callback,
            10)
        
        self.current_sub_position = None

    def callback(self, msg: JointState):
        self.current_sub_position = msg.position
        # self.get_logger().info(f'Positions: {self.current_sub_position}')



class EEPoseSubscriber(Node):
    def __init__(self):
        super().__init__('ee_pose_subscriber')
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.lookup_ee_pose)
        
    def lookup_ee_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('ur10', 'ee_link', rclpy.time.Time())
            translation = t.transform.translation
            rotation = t.transform.rotation
            print(f"End-effector pose: {translation.x*100:.2f}, {translation.y*100:.2f}, {translation.z*100:.2f}")
            print(f"End-effector pose: {rotation}")
        except Exception as error:
            self.get_logger().info(f"NO transform available: {error}")
            


if __name__ == '__main__':
    rclpy.init(args=None)
    # node = JointSubscriber()
    node = EEPoseSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
