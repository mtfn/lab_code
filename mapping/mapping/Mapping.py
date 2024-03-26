#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile
from std_msgs.msg import Float64MultiArray
from scipy.stats import norm
import pickle
from .generateFPL import *

class mappingNode(Node):
    def __init__(self):
        super().__init__("Single_R_Mapping")
        Name = self.get_namespace().split("T")[0]
        if Name=="/":
            Name = ""
        self.declare_parameter("fpl","/home/ubuntu/fpl.pcl")
        self.declare_parameter("others", "/ignoreThis")
        fiNam = self.get_parameter("fpl").get_parameter_value().string_value
        otherRobots = self.get_parameter("others").get_parameter_value().string_value.split("/")

        self.create_subscription(Float64MultiArray, Name+"/scannedPoints", self.ptsCallback, 
                                                        QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        # for a in otherRobots:
        #     self.create_subscription(Float64MultiArray, "/"+a+"/scannedPoints", self.ptsCallback, 
        #                                                 QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        # self.getMu = self.create_subscription(Float64MultiArray, Name+"/muCov", self.muCovCallback, 
        #                                                 QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        self.pubM = self.create_publisher(Float64MultiArray, Name+"/muCov", 10)

        try:
            with open(fiNam, 'rb') as handle:
            # with open('/home/saimai/Desktop/Muro lab/dgvi/fpl.pcl', 'rb') as handle:
                a = pickle.load(handle)
                self.fpoints = a[0]
                self.lscale = a[1]
            self.get_logger().info("Got fpoints from file.")
        except:
            fiNam2 = fiNam.split('fpl.pcl')[0]+'scannedPoints.txt'
            self.get_logger().info("fpl file not found, generating new fpl.pcl with "+fiNam2)
            gen(fiNam2)
            with open(fiNam, 'rb') as handle:
                a = pickle.load(handle)
                self.fpoints = a[0]
                self.lscale = a[1]
            self.get_logger().info('fpoints saved in '+fiNam)

        self.xi = 0.61
        self.d_dim = self.fpoints.shape[0] + 1

        # Subscribe to previous value of these matrices
        self.mu_update = np.zeros((0, 1, self.d_dim))  # Stores updated means
        self.cov = 5 * np.eye(self.d_dim)  # Initial covariance matrix
        self.n_omega_inv = np.linalg.inv(self.cov)  # Inverse covariance matrix

        self.error = []  # To store error values
        self.l_error = []  # To store verification error values
        
        self.get_logger().info("Waiting for points...")
        self.create_timer(3, self.postMuCov)

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
    
    # def muCovCallback(self, msg: Float64MultiArray):
    #     data = msg.data
    #     self.mu_update[-1, 0, :] = np.array(data[:(self.d_dim)]).reshape(self.d_dim)
    #     self.cov = np.array(data[(self.d_dim):]).reshape(self.d_dim,self.d_dim)
    #     # print(self.mu_update[-1, 0, :] )
        

    def ptsCallback(self, msg: Float64MultiArray):
        data = np.array(msg.data).reshape(-1,3)
        n = data.shape[0]
        X = data[:,0:2]
        y = data[:,2]
        m = self.mu_update.shape[0]
        self.mu_update = np.append(self.mu_update,np.zeros((n, 1, self.d_dim)),axis=0)
        idxa = np.random.randint(0, n-1, n)

        for t in range(m-1,m+n-1):# B is the number of data points
                
            self.mu = self.mu_update[t, 0,:] # Current mean vector
            
            idx = idxa[t-m+1]
            # Compute RBF features for the current data point
            self.Phi_X = self.feature_RBF(X[idx,:], self.fpoints, self.lscale)
            
            self.cov_phi = self.cov@self.Phi_X
            self.phi_cov_phi = self.Phi_X@self.cov_phi
            self.phi_mu = self.Phi_X@self.mu
            
            self.beta = 1 + (self.xi**2)*self.phi_cov_phi
            self.op = np.outer(self.Phi_X, self.Phi_X)
            self.xi2_beta = self.xi**2/self.beta
            self.gamma = (self.xi2_beta/(2*np.pi))**(1./2)
            self.gamma = self.gamma*np.exp(- (self.xi2_beta/2.)*(self.phi_mu*self.phi_mu))
            self.dder = self.gamma*self.op
            
            self.der = self.xi*(self.Phi_X@self.mu/(self.beta**(0.5)))
            self.der = (y[idx] - norm.cdf(self.der))*self.Phi_X # Updated derivative
            
            # Generate likelihood update
            self.n_omega_inv = self.n_omega_inv + self.dder
            self.n_sigma_ = self.cov - self.gamma*(1./(1.+self.gamma*self.phi_cov_phi))*np.outer(self.cov_phi, self.cov_phi)
            self.n_mu = self.mu + self.n_sigma_@self.der # Update mean
            self.mu_update[t + 1, 0, :] = self.n_mu.flatten()  # Store updated mean
            self.cov = self.n_sigma_
    
    def postMuCov(self):
        if self.mu_update.shape[0]==0: 
            return
        tem = Float64MultiArray()
        tem2 = np.array([])
        tem2 = np.append(tem2,self.mu_update[-1,0,:].flatten())
        tem2 = np.append(tem2,self.cov.flatten())
        tem.data = list(tem2)
        self.pubM.publish(tem)
        self.get_logger().info("Publishing latest mu and cov.")
"""
Publish mu_update[-1,0,:], cov
"""
    

def main(args=None):
    rclpy.init(args=args)
    node = mappingNode();
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown

if __name__ == '__main__':
    main()
