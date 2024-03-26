#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseWithCovariance
from rclpy.qos import ReliabilityPolicy, QoSProfile
import matplotlib.pyplot as plt
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray
from marvelmind_ros2_msgs.msg import BeaconPositionAddressed, HedgePositionAddressed

class plotPathNode(Node):
    def __init__(self):
        super().__init__("Plot_Hedgehog_Path")
        self.robot = self.get_namespace()
        if self.robot=="/":
            self.robot = "/"+input("Enter the namespace: ")
        topicName = self.robot+"/scannedPoints"
        self.robot2 = self.robot+"T"


        self.poseL = self.create_subscription(Float64MultiArray, topicName, self.callbackFn4, 
                                                    QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.get_logger().info("Waiting for points...")
            
        ## For testing Arrays message
        if '/testing' in self.robot:
            self.out = self.create_publisher(Float64MultiArray, topicName, 10)
            self.create_timer(3.0,self.sendDataFromFile)
            # pth = "/home/"
            pth = input("Path to saved points txt file: ")
            file = open(pth)
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
        plt.xlim(-3,6)
        plt.ylim(-6,3)
        plt.pause(0.01)

        

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