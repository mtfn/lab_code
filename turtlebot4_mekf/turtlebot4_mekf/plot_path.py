#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseWithCovariance
from rclpy.qos import ReliabilityPolicy, QoSProfile
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray
from marvelmind_ros2_msgs.msg import BeaconDistance, BeaconPositionAddressed, HedgePositionAddressed

class plotPathNode(Node):
    def __init__(self):
        super().__init__("Plot_Hedgehog_Path")
        topicName = self.get_namespace()
        if topicName=="/":
            topicName = input("Topic name: ")
        self.robot = "/" + topicName.split("/")[1]
        self.robot2 = self.robot+"T"

        topicType =topicName.split("/")[-1]
        
        if topicType=='pose_pub':
            self.poseL = self.create_subscription(PoseWithCovariance, topicName, self.callbackFn2, 
                                                        QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        elif topicType=='imu':
            self.poseL = self.create_subscription(Imu, topicName, self.callbackFn3, 
                                                        QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
            self.get_logger().info("Waiting Imu")
        elif topicType=='scannedPoints':
            self.poseL = self.create_subscription(Float64MultiArray, topicName, self.callbackFn4, 
                                                        QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
            self.get_logger().info("Waiting for points...")
        else:
            self.poseL = self.create_subscription(Odometry, topicName, self.callbackFn1, 
                                                        QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
            
        ## For testing Arrays message
        if self.robot=='/testing':
            self.out = self.create_publisher(Float64MultiArray, topicName, 10)
            self.create_timer(0.1,self.sendDataFromFile)
            file = open("/home/saimai/Desktop/Muro lab/scannedPoints.txt")
            lns = file.readlines()
            file.close()
            n = len(lns)
            self.n = n
            ln = []
            for i in range(n):
                ln.append(lns[i].split(","))
            self.dat = []
            for i in range(n):
                self.dat.append(float(ln[i][0]))
                self.dat.append(float(ln[i][1]))
                self.dat.append(float(ln[i][-1]))
            self.cnt = 0
            
            
        self.beacons_pos = self.create_subscription(BeaconPositionAddressed, self.robot2+"/beacons_pos_addressed", self.becPositionCallback,
                                            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        
        self.hedgehog_pos = self.create_subscription(HedgePositionAddressed, self.robot2+"/hedgehog_pos_addressed", self.hedgPositionCallback,
                                            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        
        self.inMove = self.create_publisher(Twist, self.robot+"/cmd_vel", 10)
            
        self.started = 0

        self.get_logger().info("Waiting for data from "+topicName+"...")
        self.inTr = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.inyaw = 0
        self.preX = []
        self. preY = []

        plt.axis([0,6,0,-6])
        xs = []
        ys = []
        plt.show()
        self.beacons = {12: 0., 47: 0., 66: 0., 98: 0., 99: 0}
        self.hedgehogLs = {73: [], 77: []}
        self.hedgehogs = {}
        self.stat_beacons = {}
        self.step = 0

    def becPositionCallback(self, msg: BeaconPositionAddressed):
        if msg.address in list(self.beacons.keys()): 
            self.stat_beacons[msg.address] = [
                msg.x_m, msg.y_m, msg.z_m]
            
    def hedgPositionCallback(self, msg: HedgePositionAddressed):
        if msg.address in list(self.hedgehogLs.keys()):  
            self.hedgehogs[msg.address] = [
                msg.x_m, msg.y_m, msg.z_m]

    def sendDataFromFile(self):
        if self.cnt<=self.n:
            tem = Float64MultiArray()
            self.get_logger().info("sending data")
            if self.cnt+720<=self.n:
                tem2 = [self.dat[i] for i in range(self.cnt*3,(self.cnt+720)*3)]
            else:
                tem2 = [self.dat[i] for i in range(self.cnt*3,(self.n)*3)]
            tem.data = tem2
            self.cnt+=720
            self.out.publish(tem)
        else:
            pass

    def callbackFn4(self, msg: Float64MultiArray):
        data = np.array(msg.data).reshape(-1,3)
        n = len(data)
        plt.scatter(data[:,0],data[:,1],c=data[:,2],s=1)
        for a in list(self.stat_beacons.keys()):
            plt.scatter(self.stat_beacons[a][0],self.stat_beacons[a][1], marker = "D", color = 'red', s=30)
            plt.text(self.stat_beacons[a][0]+0.2,self.stat_beacons[a][1], str(a), color = 'blue', fontsize=12)
        # for a in list(self.hedgehogs.keys()):
        #     plt.scatter(self.hedgehogs[a][0],self.hedgehogs[a][1], marker = "P", color = 'green', s=27)
        #     plt.text(self.hedgehogs[a][0]+0.2,self.hedgehogs[a][1], str(a), color = 'darkgreen', fontsize=12)
        plt.xlim(-5,6)
        plt.ylim(-6,5)
        plt.pause(0.01)

    def callbackFn1(self, msg: Odometry):
        self.pos = np.matmul(self.inTr,np.array([[msg.pose.pose.position.x],[msg.pose.pose.position.y],[1]]))
        x, y, z, w = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                      msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        num, denom = 2.0 * (w*z + x*y), 1.0 - 2.0 * (z**2 + y**2)
        self.yaw = self.repairAngValue(np.arctan2(num, denom)-self.inyaw)
        self.v = msg.twist.twist.linear.x
        self.w = msg.twist.twist.angular.z

        if not(self.started):
            self.started = True

        self.get_logger().info("Position: ("+str(self.pos[0][0])+", "+str(self.pos[1][0])+")")
        self.get_logger().info("Yaw angle: "+str(self.yaw))
        plt.cla()
        self.preX.append(self.pos[0][0])
        self.preY.append(self.pos[1][0])
        n = len(self.preX)-2
        plt.scatter(self.preX[n-60:n-21],self.preY[n-60:n-21])
        plt.scatter(self.preX[n-21:n-9],self.preY[n-21:n-9])
        plt.scatter(self.preX[n-9:n-3],self.preY[n-9:n-3])
        plt.scatter(self.preX[n-3:-1],self.preY[n-3:-1])
        plt.scatter(self.pos[0][0],self.pos[1][0])
        plt.arrow(self.pos[0][0],self.pos[1][0],0.5*np.cos(self.yaw),0.5*np.sin(self.yaw),width=0.05)
        # plt.arrow(3,3,1*np.cos(self.yaw),1*np.sin(self.yaw),width=0.05)

        for a in list(self.stat_beacons.keys()):
            plt.scatter(self.stat_beacons[a][0],self.stat_beacons[a][1], marker = "D", color = 'red', s=30)
            plt.text(self.stat_beacons[a][0]+0.2,self.stat_beacons[a][1], str(a), color = 'blue', fontsize=12)
        for a in list(self.hedgehogs.keys()):
            plt.scatter(self.hedgehogs[a][0],self.hedgehogs[a][1], marker = "P", color = 'green', s=27)
            plt.text(self.hedgehogs[a][0]+0.2,self.hedgehogs[a][1], str(a), color = 'darkgreen', fontsize=12)
        plt.xlim(-2,6)
        plt.ylim(-6,2)
        plt.pause(0.01)

    def callbackFn3(self, msg: Imu):
        self.pos = np.array([0,0])
        x, y, z, w = (msg.orientation.x, msg.orientation.y,
                        msg.orientation.z, msg.orientation.w)
        num, denom = 2.0 * (w*z + x*y), 1.0 - 2.0 * (z**2 + y**2)
        self.yaw = self.repairAngValue(np.arctan2(num, denom)-self.inyaw)

        self.get_logger().info("Yaw angle: "+str(self.yaw))
        plt.cla()
        plt.scatter(self.pos[0],self.pos[1])
        plt.arrow(3,3,1*np.cos(self.yaw),1*np.sin(self.yaw),width=0.05)
        plt.xlim(0,6)
        plt.ylim(0,-6)
        plt.pause(0.05)

    def callbackFn2(self, msg: PoseWithCovariance):
        x = msg.pose.position.x
        y = msg.pose.position.y
        plt.cla()
        self.preX.append(x)
        self.preY.append(y)
        n = len(self.preX)-2
        plt.scatter(self.preX[n-50:n-15],self.preY[n-50:n-15])
        plt.scatter(self.preX[n-15:n-9],self.preY[n-15:n-9])
        plt.scatter(self.preX[n-9:n-3],self.preY[n-9:n-3])
        plt.scatter(self.preX[n-3:-1],self.preY[n-3:-1])
        for a in list(self.stat_beacons.keys()):
            plt.scatter(self.stat_beacons[a][0],self.stat_beacons[a][1], marker = "D", color = 'red', s=30)
            plt.text(self.stat_beacons[a][0]+0.2,self.stat_beacons[a][1], str(a), color = 'blue', fontsize=12)
        for a in list(self.hedgehogs.keys()):
            plt.scatter(self.hedgehogs[a][0],self.hedgehogs[a][1], marker = "P", color = 'green', s=27)
            plt.text(self.hedgehogs[a][0]+0.2,self.hedgehogs[a][1], str(a), color = 'darkgreen', fontsize=12)
        x1, y1, z, w = (msg.pose.orientation.x, msg.pose.orientation.y,
                      msg.pose.orientation.z, msg.pose.orientation.w)
        num, denom = 2.0 * (w*z + x1*y1), 1.0 - 2.0 * (z**2 + y1**2)
        # self.yaw = self.repairAngValue(np.arctan2(num, denom)-self.inyaw)
        self.yaw = z

        plt.arrow(x,y,0.5*np.cos(self.yaw),0.5*np.sin(self.yaw),width=0.05)
        plt.scatter(x,y)
        plt.xlim(-3,6)
        plt.ylim(-6,3)
        plt.pause(0.1)
        

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


def main(args=None):
    rclpy.init(args=args)
    node = plotPathNode();
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown

if __name__ == '__main__':
    main()