import numpy as np
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped, PoseWithCovariance
from nav_msgs.msg import Odometry
from marvelmind_ros2_msgs.msg import BeaconDistance, BeaconPositionAddressed, HedgePositionAddressed
from rclpy.qos import ReliabilityPolicy, QoSProfile
from sensor_msgs.msg import Imu
import os.path
import pickle

class Ekf(object):
    """
    States:
    [0:2] position vector in intertial frame
    """

    def __init__(self, d):
        self.d = d  # State dimensions
        # number of stationary beacons, and the dimensionality of observations (Distance per beacon)
        self.d_dim = 4
        self.mu = np.zeros(d)
        self.cov = np.diag([1., 1., 1e-3, 5*1e-2])
        # Process noise covariance
        self.W = 1e-3*np.diag([1., 1., 1e-3, 5*1e-2])
        self.Sigma_z = np.eye(self.d_dim)  # Measurement noise covariance
        self.u = np.array([0., 0.])  # Default inputs
        self.x_sb = None  # Stationary beacon positions
        self.observations = {}
        self.beacons_data = {} # {beacon No.: [[beacon pos], dist_meas]}

    """
    Assume current robot position is given by the normal distribution N(mu, Sigma).
    Then, we can evaluate the plausibility of observing any distances as an argument to a Bernoulli distribution.
    Using Gaussian likelihood ratios
    """

    def acceptance(self, mu):
        """
        :param obs: Scalar distance observation received
        :param stat_pos: Known position of stationary beacon
        :param mu: Mean position of the robot
        :param Sigma: Covariance of estimated robot position
        :param Sigma_z: is the covariance associated with sensor calibration

        :param accept: Boolean variable indicating whether the data should be accepted.
        """

        """
        # Computing the maximum likelihood estimate \phi(obs| d(mu, stat_pos), Sigma_z)/ \phi(d(mu, stat_pos)| d(mu, stat_pos), Sigma_z)
        # Can be improved with hypothesis testing techniques
        """
        gamma = 0.005/self.d_dim  # Tuning parameter: Less than 1
        current_mu = np.tile(mu[:-1].flatten(), (self.d_dim, 1))
        d_ds_diff = self.observations-np.linalg.norm(current_mu-self.x_sb)
        imat = np.linalg.inv(self.Sigma_z)
        prob = np.exp(-gamma*d_ds_diff.reshape(1, -
                      1).dot(imat.dot(d_ds_diff.reshape(-1, 1)))).flatten()
        return np.random.binomial(2, prob, 1)

    def ekf_observe(self, mu, cov):
        """
        obs: Observations at any time step
        mu: Mean estimated position
        cov: Estimated position covariance 
        x_sb: True stationary beacon positions
        Sigma_z = np.eye(n)
        """
        # Generate likelihood update
        current_mu = np.tile(mu[:-1], (self.d_dim, 1))
        H_t = ((current_mu - self.x_sb).T / np.linalg.norm(self.x_sb -
               current_mu, axis=1)).T  # d_dimx3 matrix
        H_t = np.hstack((H_t, np.zeros((self.d_dim, 1))))
        K_t = (np.linalg.solve(np.dot(H_t, (np.dot(cov, H_t.T))) +
               self.Sigma_z, np.dot(H_t, cov))).T

        innovation = np.array(self.observations).flatten(
                ) - np.linalg.norm(self.x_sb - current_mu, axis=1)
        n_mu = mu + np.dot(K_t, innovation.reshape(-1, 1)).flatten()
        n_mu = n_mu.flatten()
        n_cov = np.dot((np.eye(self.d) - np.dot(K_t, H_t)), cov)
        n_cov = 0.5*(n_cov + n_cov.T)
        return n_mu, n_cov

    def ekf_predict(self, mu, cov, dt):
        """
        Turtlebot differential drive kinematics
        """
        Fk = np.array([[1., 0., 0., -self.u[0]*np.sin(mu[3])*dt],
                       [0., 1., 0., self.u[0]*np.cos(mu[3])*dt],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]])

        Bk = np.array([[np.cos(mu[3])*dt, 0],
                       [np.sin(mu[3])*dt, 0],
                       [0., 0.],
                       [0, self.u[1]*dt]])

        # Prediction step
        n_mu = np.dot(Fk, mu.reshape(-1, 1)) + \
            np.dot(Bk, self.u.reshape(-1, 1))
        n_cov = np.dot(Fk, np.dot(cov, Fk.T)) + self.W
        return (n_mu.flatten(), n_cov)


class ekf_node(Node):

    def __init__(self):
        super().__init__('ekf_node')
        self.name = ""
        self.name2 = self.get_namespace()
        if self.name2=="/":
            self.name2 = ""
        else:
            self.name = self.name2.split("T")[0]

        # self.declare_parameter("robotName", "NameSpace")
        # # self.name = self.get_parameter('robotName').get_parameter_value().string_value
        # self.name = self.get_namespace()
        # if self.name=="/":
        #     self.name = ""

        d = 4  # Working in planar domain, x,y,z, yaw theta
        self.efilter = Ekf(d) 

        self.mm_input = self.create_subscription(
            BeaconPositionAddressed,
            self.name2+'/beacons_pos_addressed',
            self.positionCallback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
            )
        
        self.dist_input = self.create_subscription(
            BeaconDistance,
            self.name2+'/beacon_raw_distance',
            self.distanceCallback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
            )
        
        self.sub_input = self.create_subscription(
            Odometry,
            self.name+'/odom',  #/cmd_vel
            self.inputCallback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
            # QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
            )
        
        self.imu_sub = self.create_subscription(
            Imu,
            self.name+'/imu',
            self.imuCallback, 
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
            )
        
        self.hedgehog_pos = self.create_subscription(HedgePositionAddressed, self.name2+"/hedgehog_pos_addressed", self.hedgPositionCallback,
                                            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        
        self.pos_pub = self.create_publisher(
            PoseWithCovariance,
            self.name+"/pose_pub",
            10
            )
        
        self.inMove = self.create_publisher(Twist, self.name+"/cmd_vel", 10)
        
        self.pose = PoseWithCovariance()

        print("Initializing...")
        self.inyaw = 0.0
        self.started = 0
        if os.path.isfile('/home/ubuntu'+self.name+'InitYaw.pcl'):
            with open('/home/ubuntu'+self.name+'InitYaw.pcl', 'rb') as handle:
                self.inyaw = pickle.load(handle)
            self.get_logger().info("Got inyaw from file, inyaw="+str(self.inyaw))
            self.still = False
        else:
            self.still = True
        self.moving = False
        self.hedgehogs = {}
        self.timer = self.create_timer(0.09, self.timerCallback)
        self.create_timer(3,self.timer2)

    def timer2(self):
        self.moving = False

    def hedgPositionCallback(self, msg: HedgePositionAddressed):
        self.hedgehogs[msg.address] = np.array([msg.x_m, msg.y_m])
        
    def positionCallback(self, stat_pos):
        if stat_pos.address not in self.efilter.beacons_data:
            self.efilter.beacons_data[stat_pos.address] = [None, None]
        
        self.efilter.beacons_data[stat_pos.address][0] = [
            stat_pos.x_m, stat_pos.y_m, stat_pos.z_m]
        
        self.efilter.x_sb = [
            v[0] for k,v in self.efilter.beacons_data.items() if v[1] != None]
        self.efilter.d_dim = len(self.efilter.x_sb)
        self.efilter.Sigma_z = np.eye(self.efilter.d_dim)

    def distanceCallback(self, dist):
        if dist.address_beacon not in self.efilter.beacons_data:
            self.efilter.beacons_data[dist.address_beacon] = [None, None]

        self.efilter.beacons_data[dist.address_beacon][1] = dist.distance_m

        self.efilter.observations = [
            v[1] for k,v in self.efilter.beacons_data.items() if v[1] != None]

    def inputCallback(self, sub_input=[0, 0]):
        if not(self.still) and self.moving:
            self.efilter.u = \
                np.array([sub_input.linear.x/20.0, sub_input.angular.z]).flatten()

    def inputCallback(self, inp: Odometry):
        if not(self.still):
            # self.efilter.u = \
            #     np.array([inp.twist.twist.linear.x, inp.twist.twist.angular.z]).flatten()
            if inp.twist.twist.linear.x>0.03:
                self.moving = True
        
    def imuCallback(self, msg: Imu):
        ori = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]

        self.pose.pose.orientation.x = ori[0]
        self.pose.pose.orientation.y = ori[1]
        self.pose.pose.orientation.z = ori[2]
        self.pose.pose.orientation.w = ori[3]

        # yaw = math.atan2(2*((ori[0]*ori[1]) + (ori[2]*ori[3])),
        #                  ori[3]**2 + ori[0]**2 - ori[1]**2 - ori[2]**2)
        psi = math.atan2(2*((ori[0]*ori[1]) + (ori[2]*ori[3])),
                         1.0-2.0*(ori[1]**2+ori[2]**2))
        yaw = psi-self.inyaw
        
        if self.still:
            if self.started==2100:
                self.inPos = np.array([self.efilter.mu[0],self.efilter.mu[1]])
                self.inyaw = yaw
                self.started+=1
            elif self.started<2100:
                self.started+=1
            else:
                cmd = Twist()
                cmd.linear.x = 0.15
                curPos = np.array([self.efilter.mu[0],self.efilter.mu[1]])
                if np.linalg.norm(curPos-self.inPos)>=0.6:
                    cmd = Twist()
                    err = {}
                    for a in self.hedgehogs.keys():
                        err[a] = np.linalg.norm(curPos-self.hedgehogs[a])
                    if (b<0.03 for b in (err.values())):
                        diff = curPos-self.inPos
                        self.inyaw -= np.arctan2(diff[1],diff[0])
                        with open('/home/ubuntu'+self.name+'InitYaw.pcl', 'wb+') as handle:
                            pickle.dump(self.inyaw, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        self.get_logger().info("Saved inyaw to file, inyaw="+str(self.inyaw))
                        self.still = False
                self.inMove.publish(cmd)
        self.pose.pose.orientation.z = yaw
        self.pose.pose.orientation.w = psi
        self.efilter.mu[3] = yaw

    def timerCallback(self):
        T = 2
        try:
            mu, cov, x_sb = self.efilter.mu, self.efilter.cov, self.efilter.x_sb
            if (x_sb is not None) and ([0., 0., 0.] not in x_sb):

                [n_mu, n_cov] = self.efilter.ekf_predict(mu, cov, 1./T)

                self.efilter.acceptance(n_mu)
                [self.efilter.mu, self.efilter.cov] = \
                    self.efilter.ekf_observe(n_mu, n_cov)

                # print('Mean estimate', self.efilter.mu, self.efilter.observations)
                # print('angle offset = '+str(self.inyaw))
        except:
            pass

        self.pose.pose.position.x = self.efilter.mu[0]
        self.pose.pose.position.y = self.efilter.mu[1]
        self.pose.pose.position.z = self.efilter.mu[2]
        self.pos_pub.publish(self.pose)

def main(args=None):
    rclpy.init(args=args)
    node = ekf_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()