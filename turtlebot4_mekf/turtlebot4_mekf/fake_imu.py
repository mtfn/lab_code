import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3
import numpy as np

class ImuPublisher(Node):

    def __init__(self):
        super().__init__('imu_publisher')
        self.publisher_1 = self.create_publisher(Imu, '/imu', 10)
        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.get_logger().info('Initialized')

    def timer_callback(self):
        rand = np.random.normal(loc=0.0, scale=0.01, size=(4,))
        imu_msg = Imu()
        imu_msg.header.frame_id = 'base_link'
        imu_msg.orientation.x = 0.
        imu_msg.orientation.y = 0.
        imu_msg.orientation.z = 0.
        imu_msg.orientation.w = 1.
        imu_msg.orientation_covariance[0] = -1.
        imu_msg.angular_velocity.x = 0.
        imu_msg.angular_velocity.y = 0.
        imu_msg.angular_velocity.z = 0.5
        imu_msg.angular_velocity_covariance = [0., 0., 0., 0., 0., 0., 0., 0., 0.05]
        imu_msg.linear_acceleration.x = 0.
        imu_msg.linear_acceleration.y = 0.
        imu_msg.linear_acceleration.z = 0.
        imu_msg.linear_acceleration_covariance = [0.001, 0., 0., 0., 0.001, 0., 0., 0., 0.001]
        self.publisher_1.publish(imu_msg)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    imu_publisher = ImuPublisher()

    rclpy.spin(imu_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()