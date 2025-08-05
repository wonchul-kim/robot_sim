import rclpy
from publisher import JointPublisher
from subscriber import JointSubscriber
from threading import Thread


class UR:
    def __init__(self):
        rclpy.init(args=None)
        self.joint_subscriber = JointSubscriber()
        self.joint_publisher = JointPublisher()
        
        
        sub_thread = Thread(target=rclpy.spin, args=(self.joint_subscriber,), daemon=True)
        sub_thread.start()
        
    
if __name__ == '__main__':
    import time 
    
    ur = UR()
    for _ in range(3):
        print(f'current_sub_position: {ur.joint_subscriber.current_sub_position}')
        time.sleep(1)
        
    ur.joint_publisher.publish_joint_positions([0.1, 0.2, 0.3, 0.1, 0.0, 0.0])
    for _ in range(3):
        print(f'current_sub_position: {ur.joint_subscriber.current_sub_position}')
        time.sleep(1)
    ur.joint_publisher.publish_joint_positions([0.5, 0.0, -0.2, 0.2, 0.1, 0.0])
    for _ in range(5):
        print(f'current_sub_position: {ur.joint_subscriber.current_sub_position}')
        time.sleep(2)

    
    
    ur.joint_publisher.destroy_node()
    ur.joint_subscriber.destroy_node()
    rclpy.shutdown()
    
    
    
    

    