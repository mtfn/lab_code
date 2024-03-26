#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile
from std_msgs.msg import Float64MultiArray, String, Header
from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import CompressedImage
from scipy.stats import norm
import matplotlib.pyplot as plt
import pickle
import random
from time import time

class plottingNode(Node):
    def __init__(self):
        super().__init__("PlotMuCov")
        Name = self.get_namespace()
        fiNam = input("Path to .pcl with fpoints, Xtest,...: ")
        self.single = bool(int(input("Single robot? (1/0) ")))
        self.full = bool(int(input("Full omega/cov? (1/0) ")))
        self.strcal = bool(int(input("Mu, omega/cov in string? (1/0) ")))
        self.img = bool(int(input("Mu, omega/cov as CompressedImage? (1/0) ")))
        if Name=="/":
            n = int(input("Number of turtlebots: "))
            a = [input("Reading mu, cov from: ")]
            for i in range(1,n):
                a.append(input("Other robot"+str(i)+": "))
            self.name = a[0]
        else:
            a = Name.split("/")[1:]
            self.name = a[0]

        self.plotMap = bool(int(input("Plot map instead of error? (1/0) ")))
        self.rviz = bool(int(input("Publish to rviz? (1/0) ")))

        self.MAP_RESOLUTION = 0.05 # m/cell
        self.MAP_SIZE_X = 11 # cells
        self.MAP_SIZE_Y = 11 # cells
        self.GRID_SIZE_X = int(self.MAP_SIZE_X / self.MAP_RESOLUTION)
        self.GRID_SIZE_Y = int(self.MAP_SIZE_Y / self.MAP_RESOLUTION)
        
        for b in a:
            self.create_subscription(Float64MultiArray, "/"+b+"/scannedPoints", self.ptsCallback, 
                                                            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
            # self.create_subscription(Float64MultiArray, "/"+b+"/muCov", self.muCovCallback, 
            #                                                 QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        if not(self.strcal):
            if not(self.img):
                self.create_subscription(Float64MultiArray, "/"+a[0]+"/muCov", self.muCovCallback, 
                                                            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
                self.get_logger().info("Waiting for normal Mu Omega")
            else:
                self.create_subscription(CompressedImage, "/"+a[0]+"/muCov", self.muCovCallback, 
                                                            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
                self.get_logger().info("Waiting for img Mu Omega")
        else:
            self.create_subscription(String, "/"+a[0]+"/muCovStr", self.muCovCallback, 
                                                        QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
            self.get_logger().info("Waiting for string Mu Omega")

        # Publisher for Occupancy Grid
        if self.rviz:
            self.pub = self.create_publisher(OccupancyGrid, "/"+self.name+"/occupancy_grid", 10)

        # with open('/home/saimai/Desktop/Muro lab/'+fiNam+'.pcl', 'rb') as handle:
        with open(fiNam, 'rb') as handle:
            a = pickle.load(handle)
            # print(a)
            self.fpoints = a[0]
            self.lscale = a[1]
            self.X_ver = a[2]
            self.y_ver = a[3]

        self.d_dim = self.fpoints.shape[0] + 1
        self.mu = np.zeros((1, self.d_dim))
        
        self.maxPt = 1000
        self.X_test = np.zeros((self.maxPt,2))

        plt.axis([0,6,0,-6])
        plt.show()

        self.error = []
        self.l_error = []
        self.initTime = time()
        self.dtime = []
        self.mus = np.array([])
        self.covs = np.array([])
        if not(self.plotMap):
            self.fil = open(input("Folder to save _muCov.pcl: ")+self.name+'_muCov.pcl', 'wb')


    def ptsCallback(self, msg: Float64MultiArray):
        data = np.array(msg.data).reshape(-1,3)
        n = data.shape[0]
        X = data[:,0:2]
        y = data[:,2]
        m = self.X_test.shape[0]
        for a in X:
            if m<=self.maxPt:
                self.X_test[random.randint(0,m-1)] = a
            else:
                self.X_test[random.randint(0,m-1)] = a

    def muCovCallback(self, msg):
        self.dtime.append(time()-self.initTime)
        data = msg.data
        if self.strcal:
            data = data.split("/")
            data = [float(d) for d in data[:-1]]
        if self.img: 
            data = np.array(msg.data)
            data2 = [float(a) for a in msg.format.split(",")]
            data = np.append((data[:(self.d_dim)].astype(np.float64)-127)/127*data2[0],
                            data[(self.d_dim):].astype(np.float64)/255*data2[1])
        self.mu = np.array(data[:(self.d_dim)]).flatten()
        if not(self.full):
            if self.single:
                cov = np.array(data[(self.d_dim):]).reshape(self.d_dim)
            else:
                self.omega = np.array(data[(self.d_dim):]).reshape(self.d_dim)
                cov = self.omega**(-1)
        else:
            omega = np.array(data[(self.d_dim):]).reshape(self.d_dim,self.d_dim)
            cov = np.linalg.inv(omega)


        y_pred = []
        self.get_logger().info("Plotting")
        for x in self.X_test:
            Phi_X = self.feature_RBF(x, self.fpoints, self.lscale)     
            if not(self.full):   
                cov_phi = cov*Phi_X
            else:
                cov_phi = cov@Phi_X
            phi_cov_phi = Phi_X@cov_phi
            y_pred.append(self.forward_model(x, self.mu, phi_cov_phi, self.fpoints, self.lscale))

        if self.plotMap:
            self.plot_predict_cat(self.X_test, y_pred)
            plt.ylim([-6,5])
            plt.xlim([-5,6])  

        if self.rviz:
            self.pub.publish(self.generate_occupancy_grid(self.X_test, y_pred))  

        else:
            y_pred2 = []
            err, lerr = 0, 0
            for i, x in enumerate(self.X_ver):
                y_pred2 = self.forward_model(x, self.mu, phi_cov_phi, self.fpoints, self.lscale)
                err += -(self.y_ver[i]*np.log(y_pred2) + (1-self.y_ver[i])*np.log(1-y_pred2))
                lerr += np.abs(self.y_ver[i] - y_pred2)
            pickle.dump([self.mu, cov, err, lerr, time()], self.fil, protocol=pickle.HIGHEST_PROTOCOL)
            self.error.append(err)
            self.l_error.append(lerr)
            print(err)
            plt.cla
            plt.plot(self.dtime,self.l_error,'bo-')
            plt.title(self.name)
            plt.pause(0.1)


    def feature_RBF(self, x, fpoints, lscale):
        """
        Type: Specify poly or RBF
        args: power of function or feature points
        """
        fpoints = np.vstack((x, fpoints)) # Create bias vector, maybe append lscale here
        nf = fpoints.shape[0]
        dist = np.linalg.norm(fpoints - np.tile(x, (nf, 1)), ord=1, axis = 1) # Order 1 norm
        return np.exp(-1*dist/(2*lscale**2))

    def sigmoid(self, x):
        return 1./(1+np.exp(-x))

    def forward_model(self, x, theta, phi_cov_phi, fpoints, lscale):
        Phi_X = self.feature_RBF(x, fpoints, lscale)
        den = (1+phi_cov_phi)**(0.5)
        return self.sigmoid(Phi_X.dot(theta)/den)
    
    def plot_predict_cat(self, X, y):
        indices = np.where(np.array(y)<=0.5)
        n_indices = np.where(np.array(y)>0.5)
        # plt.cla()
        plt.scatter(X[indices, 0], X[indices, 1], marker='^', s=1, c = 'r')
        plt.scatter(X[n_indices, 0], X[n_indices, 1], marker='o', s=1, c='b')
        plt.pause(0.1)

    def generate_occupancy_grid(self, X, y):
        # Loop through X and get the corresponding y
        grid = [0] * self.GRID_SIZE_X * self.GRID_SIZE_Y
        for i, x in enumerate(X):
            # Get the cell indices, translating them 
            x_cell = int(x[0] / self.MAP_RESOLUTION + self.GRID_SIZE_X / 2) 
            y_cell = int(x[1] / self.MAP_RESOLUTION + self.GRID_SIZE_Y / 2)
            
        # Use x, y to index through 1D array and then conver to range [0, 100]
            grid_idx = x_cell + y_cell * self.GRID_SIZE_X

            if grid_idx >= 0 and grid_idx < len(grid):
                prob = int(y[i] * 100)
                # truncate to [0, 100]
                if prob < 0:
                    prob = 0
                elif prob > 100:
                    prob = 100
                    
                grid[grid_idx] = prob

        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.info.resolution = self.MAP_RESOLUTION
        msg.info.width = self.GRID_SIZE_X
        msg.info.height = self.GRID_SIZE_Y
        
        msg.info.origin.position.x = -5.0
        msg.info.origin.position.y = -6.0
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.x = 0.0
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0

        msg.data = grid
        return msg
    

def main(args=None):
    rclpy.init(args=args)
    node = plottingNode();
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown

if __name__ == '__main__':
    main()
