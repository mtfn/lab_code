#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseWithCovariance
from rclpy.qos import ReliabilityPolicy, QoSProfile
from sensor_msgs.msg import Imu
import numpy as np
import math
from .kalman2 import Kalman
from pyquaternion import Quaternion

# class theMEKFNode(Node):
#     def ___init__(self):
#         super().__init__("new_mekf")
#         topicNamespace = self.get_namespace()
#         if topicNamespace=="/":
#             topicNamespace = "";
class theMEKFNode(Node):
    def __init__(self):
        super().__init__("MEKF_For_Orientation")
        topicNamespace = self.get_namespace()
        if topicNamespace=="/":
            print("Topic:")
            topicNamespace = input()

        self.imu_sub = self.create_subscription(Imu, topicNamespace+"/imu", self.imuCallback, 
                                                        QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.pubOut = self.create_publisher(Odometry,
            topicNamespace+"/odometry/newfiltered",
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))


        self.time_delta = 0.0005
        init_orientation = Quaternion(axis = [1, 0, 0], angle=0)

        self.kalman = Kalman(init_orientation, 1.0, 0.1, 0.1, 0.1, 0.1, 1.0)

        self.time = self.get_clock().now().nanoseconds*1e9
        self.get_logger().info("Waiting for imu data from "+topicNamespace)
        self.started = False

    def imuCallback(self, msg: Imu):
        if not(self.started):
            self.get_logger().info("Got IMU.")
            self.started = True
        temp = self.get_clock().now().nanoseconds*1e9
        # self.time_delta = temp-self.time
        self.time = temp

        gyro_measurement = np.array([msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z])
            
        measured_acc = np.array([msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z])
        # print(gyro_measurement)
        # print(measured_acc)
        self.kalman.update(gyro_measurement, measured_acc, self.time_delta)
        estimate = self.kalman.estimate
        cmd = Odometry()
        # cmd.pose.pose.orientation.x = estimate[1]
        # cmd.pose.pose.orientation.y = estimate[2]
        # cmd.pose.pose.orientation.z = estimate[3]
        # cmd.pose.pose.orientation.w = estimate[0]
        cmd.pose.pose.orientation.x = msg.orientation.x
        cmd.pose.pose.orientation.y = msg.orientation.y
        cmd.pose.pose.orientation.z = msg.orientation.z
        cmd.pose.pose.orientation.w = msg.orientation.w
        self.pubOut.publish(cmd)
        self.get_logger().info("Yaw angle ="+str(self.quatToEuler(estimate)[2]))


    def quatToEuler(self, q):
        euler = []
        x = math.atan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]*q[1] + q[2]*q[2]))
        euler.append(x)

        x = math.asin(2*(q[0]*q[2] - q[3]*q[1]))
        euler.append(x)

        x = math.atan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]*q[2] + q[3]*q[3]))
        euler.append(x)

        return euler

    def quatListToEulerArrays(self, qs):
        euler = np.ndarray(shape=(3, len(qs)), dtype=float)

        for (i, q) in enumerate(qs):
            e = self.quatToEuler(q)
            euler[0, i] = e[0]
            euler[1, i] = e[1]
            euler[2, i] = e[2]

        return euler

    def eulerError(self, estimate, truth):
        return np.minimum(np.minimum(np.abs(estimate - truth), np.abs(2*math.pi + estimate - truth)),
                                    np.abs(-2*math.pi + estimate - truth))

    def eulerArraysToErrorArrays(self, estimate, truth):
        errors = []
        for i in range(3):
            errors.append(self.eulerError(estimate[i], truth[i]))
        return errors

    def quatListToErrorArrays(self, estimate, truth):
        return self.eulerArraysToErrorArrays(self.quatListToEulerArrays(estimate), self.quatListToEulerArrays(truth))

    def rmse_euler(self, estimate, truth):
        def rmse(vec1, vec2):
            return np.sqrt(np.mean((vec1 - vec2)**2))

        return[rmse(estimate[0], truth[0]), 
            rmse(estimate[1], truth[1]),
            rmse(estimate[2], truth[2])]
    

def main(args=None):
    rclpy.init(args=args)
    node = theMEKFNode();
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown

if __name__ == '__main__':
    main()
