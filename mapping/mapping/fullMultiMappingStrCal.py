#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile
from std_msgs.msg import Float64MultiArray, String
from scipy.stats import norm
import scipy
import scipy.sparse.linalg as sla
import pickle
import random

class fullMulMappingStrCalNode(Node):
    def __init__(self):
        super().__init__("Full_Multi_Mapping")
        Name = self.get_namespace()
        self.declare_parameter("fpl","fpl")
        fiNam = self.get_parameter("fpl").get_parameter_value().string_value
        if Name=="/":
            print("Name_i and Name_j:")
            Name_i = "/"+input()
            Name_j = "/"+input()
        else:
            a = Name.split("/")
            Name_i = "/"+a[2]
            Name_j = "/"+a[1]
            print("self: "+Name_i)
            print("pre: "+Name_j)

        self.poseL = self.create_subscription(Float64MultiArray, Name_i+"/scannedPoints", self.ptsCallback, 
                                                        QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.get_logger().info("Waiting for points...")
        
        self.create_subscription(String, Name_j+"/muCovStr", self.muCovStrCallback, 
                                                        QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        self.pubM2 = self.create_publisher(String, Name_i+"/muCovStr", 10)

        with open('/home/saimai/Desktop/Muro lab/'+fiNam+'.pcl', 'rb') as handle:
            a = pickle.load(handle)
            self.fpoints = a[0]
            self.lscale = a[1]

        self.xi = 0.61
        self.d_dim = self.fpoints.shape[0] + 1

        # Subscribe to previous value of these matrices
        self.n = 2
        self.lik_factor = 0.5
        self.mu_update_i = np.zeros((0, 1, self.d_dim))  # Stores updated means
        self.mu_update_j = np.zeros((1, 1, self.d_dim))

        alfa = 5.
        self.cov = alfa * np.eye(self.d_dim)  # Initial covariance matrix
        self.n_omega_i = np.linalg.inv(self.cov) 
        self.n_omega_j = np.linalg.inv(self.cov) 
        self.omega_i = (1./alfa)*np.eye(self.d_dim)
        self.omega_j = (1./alfa)*np.eye(self.d_dim)
        
        self.create_timer(3, self.postMuCov)
        self.started = 0

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
    
    def der_dder_dsig(self, y, cov_phi, Phi_X, mu):
        """
        Return the first and second derivatives wrt xi, and the first derivative wrt cov_phi and mu
        """
        #Dot products of cov_phi and mu with Phi_X
        phi_cov_phi = Phi_X@cov_phi
        phi_mu = Phi_X@mu

        op = np.outer(Phi_X, Phi_X) # Calculate outer product of Phi_X with itself
        # Calculate parameters for second derivative
        beta = 1 + (self.xi**2)*phi_cov_phi
        xi2_beta = self.xi**2/beta
        gamma = (xi2_beta/(2*np.pi))**(1./2)
        gamma = gamma*np.exp(- (xi2_beta/2.)*(phi_mu*phi_mu))
        dder = gamma*op # Calculate second derivative 

        der = self.xi*(phi_mu/(beta**(0.5))) # Calculate first derivative
        der_coeff = (y - norm.cdf(der)) # Calculate derivative coefficient for first derivative
        der = der_coeff*Phi_X

        dsig = der_coeff*cov_phi - gamma*(1./(1.+gamma*phi_cov_phi))*(cov_phi@der)*cov_phi
        return der, dder, dsig

    def muCovStrCallback(self, msg: String):
        data = msg.data
        data = data.split("/")
        data = [float(d) for d in data[:-1]]
        self.mu_update_j[0,0,:] = np.array(data[:(self.d_dim)]).flatten()
        self.omega_j = np.array(data[(self.d_dim):]).reshape(self.d_dim,self.d_dim)

        if self.mu_update_i.shape[0]!=0:
            self.mu_update_i[-1, 0,:] = 1/30*(self.omega_i@self.mu_update_i[-1, 0,:])+29/30*(self.omega_j@self.mu_update_j[-1,0,:])
            self.omega_i = 1/30*self.omega_i+29/30*self.omega_j
            self.cov = np.linalg.inv(self.omega_i)

            self.mu_update_i[-1, 0,:] = self.cov@self.mu_update_i[-1, 0,:]

    def ptsCallback(self, msg: Float64MultiArray):
        data = np.array(msg.data).reshape(-1,3)
        n = data.shape[0]
        X = data[:,0:2]
        y = data[:,2]
        m = self.mu_update_i.shape[0]
        # if self.mu_update_i.shape[0]!=0:
        #     self.mu_update_i[-1, 0,:] = 1/2*(self.omega_i@self.mu_update_i[-1, 0,:])+1/2*(self.omega_j@self.mu_update_j[-1,0,:])
        #     self.omega_i = 1/2*self.omega_i+1/2*self.omega_j
        #     self.cov = np.linalg.inv(self.omega_i)

        #     self.mu_update_i[-1, 0,:] = self.cov@self.mu_update_i[-1, 0,:]
        self.mu_update_i = np.append(self.mu_update_i,np.zeros((n, 1, self.d_dim)),axis=0)
        if self.started <= 21:
            self.started += 1
            self.postMuCov()

        for t in range(m-1,m+n-1):# B is the number of data points
            
            mu = self.mu_update_i[t-m, 0,:] # Current mean vector
            
            # Compute RBF features for the current data point
            Phi_X = self.feature_RBF(X[t-m,:], self.fpoints, self.lscale)

            cov_phi = self.cov@Phi_X
            phi_cov_phi = Phi_X@cov_phi
            phi_mu = Phi_X@mu
            
            beta = 1 + (self.xi**2)*phi_cov_phi
            op = np.outer(Phi_X, Phi_X)
            xi2_beta = self.xi**2/beta
            gamma = (xi2_beta/(2*np.pi))**(1./2)
            gamma = gamma*np.exp(- (xi2_beta/2.)*(phi_mu*phi_mu))
            dder = gamma*op
            
            der = self.xi*(Phi_X@mu/(beta**(0.5)))
            der = (y[t-m] - norm.cdf(der))*Phi_X # Updated derivative

            # omega_i = scipy.sparse.csc_matrix(omega_i).tocsc()
            # solved = sla.spsolve(omega_i, np.vstack([mu, Phi_X]).T )
            # mu = solved[:,0]
            # cov_phi = solved[:,1]

            # der, dder, dsig = self.der_dder_dsig(y[t-m], cov_phi, Phi_X, mu)
            
            # Generate likelihood update
            self.n_omega_i = self.n_omega_i + self.lik_factor*dder
            # n_mu = mu + self.lik_factor*dsig # Update mean
            # self.mu_update_i[t + 1, 0, :] = n_mu.flatten()  # Store updated mean
            n_sigma_ = self.cov - gamma*(1./(1.+gamma*phi_cov_phi))*np.outer(cov_phi, cov_phi)
            n_mu = mu + n_sigma_@der # Update mean
            self.mu_update_i[t + 1, 0, :] = n_mu.flatten()  # Store updated mean
            self.cov = n_sigma_
        self.omega_i = np.linalg.inv(self.cov)
    
    def postMuCov(self):
        if self.mu_update_i.shape[0]==0: 
            return
        # self.mu_update_i[-1,0,:] = [random.gauss(6.1,23.9) for a in range(281)]
        tem = String()
        tem2 = ""
        tem2 += self.listToString(self.mu_update_i[-1,0,:].flatten())
        tem2 += self.listToString(self.omega_i.flatten())
        tem.data = tem2
        self.pubM2.publish(tem)

    def listToString(self, s):
        str1 = ""
        for ele in s:
            if ele!=0.0:
                str1 += str(round(ele, -int(np.floor(np.log10(abs(ele))))+8))+"/"
            else:
                str1 += str(round(ele, 1))+"/"
        return str1
    

def main(args=None):
    rclpy.init(args=args)
    node = fullMulMappingStrCalNode();
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown

if __name__ == '__main__':
    main()
