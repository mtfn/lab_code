import rclpy
from rclpy.node import Node
from marvelmind_ros2_msgs.msg import BeaconDistance, BeaconPositionAddressed
import numpy as np

class BeaconPublisher(Node):

    def __init__(self):
        super().__init__('beacon_publisher')
        self.name = self.get_namespace()
        if self.name=="/":
            self.name = ""
        self.publisher_1 = self.create_publisher(BeaconPositionAddressed, self.name+"/beacons_pos_addressed", 10)
        self.publisher_2 = self.create_publisher(BeaconDistance, self.name+"/beacon_raw_distance", 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.get_logger().info('Initialized')
        self.i
        self.hedges = [1,2,3,4]
        self.xs = [1.2, 4.6, 1.5, 3.4, 1]
        self.ys = [2.2, 3.6, 4.5, 2.4, 1]
        self.zs = [5.2, 2.6, 3.5, 1.4, 1]

    def timer_callback(self):
        self.i += 1
        rand = np.random.normal(loc=0.0, scale=0.01, size=(4,))
        msg1 = BeaconPositionAddressed()
        msg1.address = self.hedges[self.i-1]
        msg1.x_m = self.xs[self.i-1] + rand[0]
        msg1.y_m = self.ys[self.i-1] + rand[1]
        msg1.z_m = self.zs[self.i-1]+ rand[2]
        self.publisher_1.publish(msg1)

        msg2 = BeaconDistance()
        msg2.address_hedge = 8
        msg2.address_beacon = self.hedges[self.i-1]
        msg2.distance_m = np.sqrt(msg1.x_m**2 + msg1.y_m**2 + msg1.z_m**2) + rand[3]
        self.publisher_2.publish(msg2)
        
        
        if self.i == len(self.hedges):
            self.i = 0


def main(args=None):
    rclpy.init(args=args)

    beacon_publisher = BeaconPublisher()

    rclpy.spin(beacon_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
