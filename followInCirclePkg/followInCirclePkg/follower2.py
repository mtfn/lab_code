#!/usr/bin/env python
import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rclpy.qos import ReliabilityPolicy, QoSProfile
from .tools import *

class follower2Node(Node):
    def __init__(self):
        super().__init__("Cyclic_Pursuit_R3")
        global namespace3, cmd1
        cmd1 = Twist()
        namespace3 = self.get_namespace()
        if namespace3=="/":
            print("namespace:")
            namespace3 = input()
        self.publ_r3_ = self.create_publisher(Twist, namespace3+"/cmd_vel", 10)
        self.publ_r3_pose_ = self.create_publisher(Odometry, "/cyclic_putsuit/r3pose", 10)

        self.dpose_r3_ = self.create_subscription(Odometry, namespace3+"/odom", self.o3_callback, 
                                                  QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.modified_dpose_r2_ = self.create_subscription(Odometry, "/cyclic_putsuit/r2pose", self.r2_pose_callback, 
                                                  QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.modified_dpose_r1_ = self.create_subscription(Odometry, "/cyclic_putsuit/r1pose", self.r1_pose_callback, 
                                                  QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        self.radius = 2*12*2.54/100   ##meters.
        self.center = np.array([0,self.radius])

        self.inyaw = 0
        self.inTr = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.angleSeparation = 2*np.pi/3
        self.foundR2 = False
        self.foundR1 = False
        self.pos2 = np.array([0,0])
        self.dt = 0.06
        self.maxSpeed = 0.18
        self.normalSpeed = 0.12
        self.maintainDist = 0.2
        self.lincon = PID(Kp = 0.08, Ki = 0.01, Kd = 0.005, max = self.maxSpeed-self.normalSpeed, min = -0.12, dt = self.dt)
        self.angcon = PID(Kp = 0.3, Ki = 0.05, Kd = 0.1, max = np.pi, min = -np.pi, dt = self.dt)
        self.started = False
        self.gotOrigin = False
        self.get_logger().info("Waiting for odometry data from "+namespace3+"...")
    
    def r1_pose_callback(self, msg: Odometry):
        self.r1ang = msg.pose.pose.orientation.z
        self.pos1 = np.array([msg.pose.pose.position.x,msg.pose.pose.position.y])
        if not(self.foundR1):
            self.get_logger().info("detected R1")
        self.foundR1 = True

    def r2_pose_callback(self, msg: Odometry):
        self.r2ang = msg.pose.pose.orientation.z
        self.pos2 = np.array([msg.pose.pose.position.x,msg.pose.pose.position.y])
        if not(self.foundR2):
            self.get_logger().info("detected R2")
        self.foundR2 = True

    def o3_callback(self, msg: Odometry):
        self.pos3 = np.matmul(self.inTr,np.array([[msg.pose.pose.position.x],[msg.pose.pose.position.y],[1]]))
        x, y, z, w = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                      msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        num, denom = 2.0 * (w*z + x*y), 1.0 - 2.0 * (z**2 + y**2)
        self.yaw3 = self.repairAngValue(np.arctan2(num, denom)-self.inyaw)
        self.v = msg.twist.twist.linear.x
        self.w = msg.twist.twist.angular.z

        if not(self.started):
            self.get_logger().info("detected "+namespace3)
            print("Set offsets (angle,x,y)")
            a = float(input());
            b = float(input());
            c = float(input());
            phi = self.repairAngValue(self.yaw3+a/180*np.pi)
            self.inyaw = self.repairAngValue(self.yaw3+a/180*np.pi)
            T1 = np.array([[1,0,-self.pos3[0][0]],[0,1,-self.pos3[1][0]],[0,0,1]])
            R1 = np.array([[np.cos(phi),np.sin(phi),0],[-np.sin(phi),np.cos(phi),0],[0,0,1]])
            T2 = np.array([[1,0,b*self.radius],[0,1,c*self.radius],[0,0,1]])
            self.inTr = np.matmul(R1,T1)
            self.inTr = np.matmul(T2,self.inTr)
            self.started = True   
        else:     
            self.pos3 = np.array([self.pos3[0][0],self.pos3[1][0]])
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
        if self.foundR2 and self.foundR1:
            distConst = 0.1
            p3a = self.getAngle(self.pos3,self.center)
            p3 = self.pos3-self.center
            p2 = self.pos2-self.center
            p1 = self.pos1-self.center
            if np.cross(p3,p2)<=0 or np.cross(p1,p3)<=0:
                radius = self.radius
            else:
                radius = self.radius
            closestPoint = self.center-np.array([radius*np.cos(p3a),radius*np.sin(p3a)])
            distanceToClosest = self.getDistance(self.pos3,closestPoint)
            perpenAng = self.getAngle(self.pos3,closestPoint)

            curr_pose = Odometry()
            curr_pose.pose.pose.orientation.z = self.getAngle(self.center,self.pos3)
            curr_pose.pose.pose.position.x = self.pos3[0]
            curr_pose.pose.pose.position.y = self.pos3[1]
            self.publ_r3_pose_.publish(curr_pose)

            if distanceToClosest<distConst:
                tempTargetOnTangentLine = closestPoint+np.array([0.15*np.cos(p3a-np.pi/2),0.15*np.sin(p3a-np.pi/2)])
                direction = self.getAngle(self.pos3,tempTargetOnTangentLine)
                change = self.repairAngValue(perpenAng - direction)
                change = 0.15*change/distConst*abs(distanceToClosest)

                newRefAng = direction + change
                angleErr = newRefAng-self.yaw3
            else:
                angleErr = perpenAng - self.yaw3
            speed  = self.normalSpeed

        # if self.foundR2 and self.foundR1:
            self.get_logger().info("Found R2!")
            tempA = self.r1ang-self.angleSeparation*2
            want_p = np.array([radius*np.cos(tempA),radius*np.sin(tempA)])
            linErr = np.sign(np.cross(p3,want_p))*self.getDistance(p3,want_p)

            speed = self.normalSpeed+self.lincon.uVal(linErr)*2+(radius-self.radius)*2
            self.get_logger().info(str(linErr))
            self.get_logger().info("speed = "+str(speed))
        else:
            speed=0.0
            angleErr=0.0


        cmd1.linear.x = speed
        cmd1.angular.z = self.angcon.uVal(self.repairAngValue(angleErr))

        self.publ_r3_.publish(cmd1)


def main(args=None):
    rclpy.init(args=args)
    node = follower2Node();
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown


if __name__ == '__main__':
    main()
