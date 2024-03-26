#!/usr/bin/env python
import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rclpy.qos import ReliabilityPolicy, QoSProfile
from .tools import *

class goInCircleNode(Node):
    def __init__(self):
        super().__init__("Cyclic_Pursuit_R1")
        global namespace1, cmd1
        cmd1 = Twist()
        namespace1 = self.get_namespace()
        self.declare_parameter('useBeacons','False')
        tempStr = self.get_parameter('useBeacons').get_parameter_value().string_value
        useBeacons = eval(tempStr)
        if namespace1=="/":
            print("namespace:")
            namespace1 = input()
        self.publ_r1_ = self.create_publisher(Twist, namespace1+"/cmd_vel", 10)
        self.publ_r1_pose_ = self.create_publisher(Odometry, "/cyclic_putsuit/r1pose", 10)

        if useBeacons:
            self.dpose_r1_ = self.create_subscription(Odometry, namespace1+"/odometry/filtered", self.o1_callback, 
                                                    QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        else:
            self.dpose_r1_ = self.create_subscription(Odometry, namespace1+"/odom", self.o1_callback, 
                                                    QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        self.radius = 2*12*2.54/100   ##meters.
        self.center = np.array([0,self.radius])

        self.inyaw = 0
        self.inTr = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.dt = 0.06
        self.maxSpeed = 0.12
        self.lincon = PID(Kp = 0.08, Ki = 0.01, Kd = 0.005, max = self.maxSpeed, min = 0.0, dt = self.dt)
        self.angcon = PID(Kp = 0.3, Ki = 0.05, Kd = 0.1, max = np.pi, min = -np.pi, dt = self.dt)
        self.started = False
        self.atCircle = False
        self.get_logger().info("Waiting for odometry data from "+namespace1+"...")

    def o1_callback(self, msg: Odometry):
        self.pos1 = np.matmul(self.inTr,np.array([[msg.pose.pose.position.x],[msg.pose.pose.position.y],[1]]))
        x, y, z, w = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                      msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        num, denom = 2.0 * (w*z + x*y), 1.0 - 2.0 * (z**2 + y**2)
        self.yaw1 = self.repairAngValue(np.arctan2(num, denom)-self.inyaw)
        self.v = msg.twist.twist.linear.x
        self.w = msg.twist.twist.angular.z

        if not(self.started):
            self.get_logger().info("detected "+namespace1)
            print("Set offsets (angle,x,y)")
            a = float(input());
            b = float(input());
            c = float(input());
            phi = self.repairAngValue(self.yaw1+a/180*np.pi)
            self.inyaw = self.repairAngValue(self.yaw1+a/180*np.pi)
            T1 = np.array([[1,0,-self.pos1[0][0]],[0,1,-self.pos1[1][0]],[0,0,1]])
            R1 = np.array([[np.cos(phi),np.sin(phi),0],[-np.sin(phi),np.cos(phi),0],[0,0,1]])
            T2 = np.array([[1,0,b*self.radius],[0,1,c*self.radius],[0,0,1]])
            self.inTr = np.matmul(R1,T1)
            self.inTr = np.matmul(T2,self.inTr)
            self.pos1 = np.matmul(self.inTr,self.pos1)
            self.get_logger().info("loc=("+str(self.pos1[0][0])+","+str(self.pos1[1][0])+")")
            self.get_logger().info("cen=("+str(self.center[0])+","+str(self.center[1])+")")
            self.started = True     
        else: 
            self.pos1 = np.array([self.pos1[0][0],self.pos1[1][0]])
            self.controller()

    def getAngle(self, v1 ,v2):
        V = v2-v1
        ang = np.arctan2(V[1],V[0])
        return ang
    def getDistance(self, v1, v2):
        V = v2-v1
        dis = np.linalg.norm(V)
        return dis
    def repairAngValue(self, ang):
        ang2 = ang
        while abs(ang2)>np.pi:
            if ang>np.pi:
                ang2 = ang-2*np.pi
            elif ang<-np.pi:
                ang2 = ang+2*np.pi
        return ang2

    def controller(self):
        distConst = 0.1
        p1a = self.getAngle(self.pos1,self.center)
        closestPoint = self.center-np.array([self.radius*np.cos(p1a),self.radius*np.sin(p1a)])
        distanceToClosest = self.getDistance(self.pos1,closestPoint)
        perpenAng = self.getAngle(self.pos1,closestPoint)

        curr_pose = Odometry()
        curr_pose.pose.pose.orientation.z = self.getAngle(self.center,self.pos1)
        curr_pose.pose.pose.position.x = self.pos1[0]
        curr_pose.pose.pose.position.y = self.pos1[1]
        self.publ_r1_pose_.publish(curr_pose)

        if distanceToClosest<distConst:
            tempTargetOnTangentLine = closestPoint+np.array([0.15*np.cos(p1a-np.pi/2),0.15*np.sin(p1a-np.pi/2)])
            direction = self.getAngle(self.pos1,tempTargetOnTangentLine)
            change = self.repairAngValue(perpenAng - direction)
            change = 0.15*change/distConst*abs(distanceToClosest)

            newRefAng = direction + change
            angleErr = newRefAng-self.yaw1
        else:
            angleErr = perpenAng - self.yaw1
        speed  = self.maxSpeed

        # if not(self.atCircle) and angleErr==perpenAng - self.yaw1:
        #     speed = 0.0
        # else:
        #     self.atCircle = True

        cmd1.linear.x = speed
        cmd1.angular.z = self.angcon.uVal(self.repairAngValue(angleErr))

        self.publ_r1_.publish(cmd1)

class JustGoNode(Node):
    def __init__(self):
        namespace1 = "catalina"
        self.publ = self.create_publisher(Twist, "/"+namespace1+"/cmd_vel", 10)
        self.timer_ = self.create_timer(0.5, self.just_go)
    def just_go(self):
        msg = Twist()
        msg.linear.x = 0.12
        msg.angular.z = msg.linear.x/0.6
        self.publ.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = goInCircleNode();
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown

def gogogo(args=None):
    rclpy.init(args=args)
    node = JustGoNode();
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown


if __name__ == '__main__':
    main()
if __name__ == '__gogogo__':
    gogogo()
