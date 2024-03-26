#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseWithCovariance
from rclpy.qos import ReliabilityPolicy, QoSProfile
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
from .scan_reader_func import *
# from .tools import followPath

class lidar2MMNode(Node):
    def __init__(self):
        super().__init__("Lidar_to_global_frame")
        self.robot = self.get_namespace()
        self.declare_parameter("speed",0.07)
        self.declare_parameter("distance",6.0)
        self.xsp = self.get_parameter("speed").get_parameter_value().double_value
        self.range = self.get_parameter("distance").get_parameter_value().double_value
        self.get_logger().info('Speed = '+str(self.xsp)+'m/s, Distance = '+str(self.range)+'ft')
        self.dis = self.range*12*2.54/100
        if self.xsp != 0.0:
            self.range = int(np.floor(self.dis/self.xsp*2))
        if self.robot=="/":
            self.robot = ""
        else:
            a = self.robot.split("/")
            self.robot = "/"+a[-1]
            self.robot = self.robot.split("T")[0]
        
        self.create_subscription(PoseWithCovariance, self.robot+"/pose_pub", self.callbackFn2, 
                                                    QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        
        self.create_subscription(Odometry, self.robot+"/est_pose_mekf", self.callbackFn3, 
                                                    QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        
        self.lidar = self.create_subscription(LaserScan, self.robot+"/scan", self.scanCallback, 
                                                    QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        
        self.move = self.create_publisher(Twist, self.robot+"/cmd_vel", 10)

        self.out = self.create_publisher(Float64MultiArray, self.robot+"/scannedPoints", 6000)

        self.get_logger().info("Waiting for data from "+self.robot+"...")
        self.inTr = np.array([[1,0,0,],[0,1,0,],[0,0,1,]])
        self.pose = np.array([0.0,0.0,0.0])
        self.yaw = 0.0
        self.psi = 0.0
        self.started = 0
        self.scan = False
        self.wait = 12
        self.scanned = 0
        self.file = open("/home/ubuntu/scannedPoints.txt","w")
        # self.file = open("/home/saimai/Desktop/Muro lab/sP123456.txt","w")
        self.dt = 0.5
        self.create_timer(self.dt, self.timerCallback)
        self.gotPose = False
        self.turning = False
        self.pastHfTurn = False
        self.inTurnAng = 0.0
        self.startPose = np.array([-1.0,0.0])
        self.turnPose = np.array([1.0,0.0])
        # self.motion = followPath(self.move, self.xsp, np.array([self.pose[0], self.pose[1]]), self.yaw)
        self.lasterr = 0.0

    def timerCallback(self):
        if self.started<42:  #36
            self.started+=1
        else:
            self.started+=1
            cmd = Twist()
            cmd.linear.x = self.xsp
            pos = np.array([self.pose[0], self.pose[1]])

            
            # print(self.startPose)
            # print(self.turnPose)
            # print(err1)
            if (self.xsp!=0.0) and ((self.started-42)%(self.range) != 0) and not(self.turning):
                err1 = np.cross([np.cos(self.yaw), np.sin(self.yaw)],self.turnPose-pos)
                # if abs(self.getDistance(self.startPose,pos)*np.sin(err1))<=0.06:
                #     err1 = self.repairAngValue(self.getAngle(self.startPose,self.turnPose)-self.getAngle(self.startPose,pos))
                cmd.angular.z = err1/6+(err1-self.lasterr)/self.dt/15
                self.lasterr = err1
            if (self.xsp != 0.0) and (np.linalg.norm(self.turnPose-pos) <= 0.15 or self.turning):
                #self.get_logger().info("stopping to turn. tn="+str(self.started-42))
                self.started -= 1
                if not(self.turning):
                    self.inTurnAng = self.yaw
                    self.pastHfTurn = False
                    self.turning = True
                    tem = self.turnPose
                    self.turnPose = self.startPose
                    self.startPose = tem
                    self.lasterr = 0.0
                # err = self.repairAngValue(self.getAngle(self.startPose,self.turnPose))-self.repairAngValue(self.yaw)
                err = np.cross([np.cos(self.yaw), np.sin(self.yaw)],self.turnPose-pos)
                cmd.linear.x = 0.0
                cmd.angular.z = err/3+(err-self.lasterr)/self.dt/18
                self.get_logger().info("err = "+str(err))
                if abs(self.lasterr)>abs(err) or self.pastHfTurn:
                    self.pastHfTurn = True
                    self.get_logger().info("Turned half!")
                    if abs(err)<=0.21:
                        self.turning = False
                        self.started += 1
                elif abs(err)<=1.0:
                    cmd.angular.z = err/abs(err)*np.pi/6
                self.lasterr = err
            elif self.started%6==0:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.started -= 1
                self.wait -= 1
                if self.wait==0 and not(self.scan):
                    self.scan = True
            self.move.publish(cmd)

    def scanCallback(self, msg: LaserScan):
        dotsCount = 3
        if self.scan and self.gotPose:
            if self.started<=70:
                self.get_logger().info("Got Lidar Data!")
                # print(self.pose)
            if self.gotPose and self.pose[0] != 0.0:
                phi = self.yaw+np.pi/2
                R = np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]])
                T = np.array([[[1,0,self.pose[0]],[0,1,self.pose[1]],[0,0,1]]])
                Tr = np.matmul(T,R)
                coords = np.asarray(get_coords(msg, dotsCount))
                coords = np.transpose(coords)
                coords = np.matmul(Tr,coords)
                coords = np.transpose(coords)
                tem = Float64MultiArray()
                tem2 = []
                for a in range(len(coords)):
                    self.file.write(str(coords[a][0][0])+","+str(coords[a][1][0])+",")
                    if (a+1)%(dotsCount+1)==0:
                        self.file.write("1\n")
                        lb = 1.0
                    else:
                        self.file.write("0\n")
                        lb = 0.0
                    tem2.append(float(coords[a][0][0]))
                    tem2.append(float(coords[a][1][0]))
                    tem2.append(lb)
                tem.data = tem2
                self.out.publish(tem)
                self.scanned += 1
                if self.scanned>=3:
                    self.started += 1
                    self.scanned = 0
                    self.wait = 12
                    self.scan = False

    def callbackFn2(self, msg: PoseWithCovariance):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        self.pose[0] = x
        self.pose[1] = y
        self.pose[2] = z
        self.yaw = msg.pose.orientation.z
        self.psi = msg.pose.orientation.w

        if not(self.gotPose) and self.yaw != 0.0:
            self.gotPose = True
            self.get_logger().info("Got Position")
            self.startPose = np.array([x,y])
            self.turnPose = np.array([x+self.dis*np.cos(self.yaw), y+self.dis*np.sin(self.yaw)])

    def callbackFn3(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        self.pose[0] = x
        self.pose[1] = y
        self.pose[2] = z
        ox = msg.pose.pose.orientation.x
        oy = msg.pose.pose.orientation.y
        oz = msg.pose.pose.orientation.z
        ow = msg.pose.pose.orientation.w
        self.yaw = self.quat2yaw(quat = [ox,oy,oz,ow])
        if not(self.gotPose) and self.yaw != 0.0:
            self.gotPose = True
            self.get_logger().info("Got Position")
            self.startPose = np.array([x,y])
            self.turnPose = np.array([x+self.dis*np.cos(self.yaw), y+self.dis*np.sin(self.yaw)])
    
    def repairAngValue(self, ang):
        ang2 = ang
        while abs(ang2)>np.pi:
            if ang>np.pi:
                ang2 = ang-2*np.pi
            elif ang<-np.pi:
                ang2 = ang+2*np.pi
        return ang2
    
    def getAngle(self, v1 ,v2):
        V = v2-v1
        ang = np.arctan2(V[1],V[0])
        return ang
    def getDistance(self, v1, v2):
        V = v2-v1
        dis = np.linalg.norm(V)
        return dis
    
    def quat2yaw(self, quat):
        [x, y, z, w] = [quat[0], quat[1], quat[2], quat[3]]
        num, denom = 2.0 * (w*z + x*y), 1.0 - 2.0 * (z*z + y*y)
        yaw = np.arctan2(num, denom)
        return yaw

def main(args=None):
    rclpy.init(args=args)
    node = lidar2MMNode();
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown

if __name__ == '__main__':
    main()
