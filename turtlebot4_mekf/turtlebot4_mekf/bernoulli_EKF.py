#!/usr/bin/env python3
import numpy as np
from rclpy.node import Node
import rclpy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped, PoseWithCovariance
from rclpy.qos import ReliabilityPolicy, QoSProfile
# from turtlebot_deployment.msg import PoseWithName
from marvelmind_ros2_msgs.msg import BeaconDistance, BeaconPositionAddressed ##beacon_distance, beacon_pos_a
# Depends on turtlebot_deployment package msg PoseWithName.
# string name
# geometry_msgs/Pose pose

### Use "ros2 run turtlebot4_mekf Bernoulli_EKF --ros-args robotName:={THE_NAMESPACE}" to run.
### e.g. ros2 run turtlebot4_mekf Bernoulli_EKF --ros-args robotName:=yosemite

class Ekf(object):
    """
    States:
    [0:2] position vector in intertial frame
    """

    def __init__(self, d):
        self.d = d  # State dimensions
        self.stat_beacons = {}
        # number of stationary beacons, and the dimensionality of observations (Distance per beacon)
        d_dim = 4
        self.mu = np.zeros(d)
        self.cov = np.diag([1., 1., 1e-3, 5*1e-2])
        # Process noise covariance
        self.W = 1e-3*np.diag([1., 1., 1e-3, 5*1e-2])
        self.Sigma_z = np.eye(d_dim)  # Measurement noise covariance
        self.u = np.array([0., 0.])  # Default inputs
        self.x_sb = None  # Stationary beacon positions
        self.observations = {12: 0., 47: 0., 66: 0., 98: 0.}
        self.my_addresekf_accepts = 0

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
        d_ds_diff = list(self.observations.values())-np.linalg.norm(current_mu-self.x_sb) ##made dict_values into list [Saimai]
        imat = np.linalg.inv(self.Sigma_z)
        # a = input();
        prob = np.exp(-gamma*d_ds_diff.reshape(1, -
                      1).dot(imat.dot(d_ds_diff.reshape(-1, 1)))).flatten()
        return np.random.binomial(2, prob, 1)

    """
    Mahalanobis distance for acceptance.
    Implement acceptance for individual beacons.
    """

    def m_acceptance(self, mu):
        """
        :param obs: Scalar distance observation received
        :param stat_pos: Known position of stationary beacon
        :param mu: Mean position of the robot
        :param Sigma: Covariance of estimated robot position
        :param Sigma_z: is the covariance associated with sensor calibration

        :param accept: Boolean variable indicating whether the data should be accepted.
        """
        # Computing the Mahalanobis distance

        std = 3
        current_mu = np.tile(mu[:-1].flatten(), (self.d_dim, 1))
        d_ds_diff = list(self.observations.values())-np.linalg.norm(current_mu-self.x_sb)  ##made dict_values into list [Saimai]
        self.Sigma_z = np.eye(self.d_dim)
        imat = np.linalg.inv(self.Sigma_z)
        dist_dev = np.sqrt(d_ds_diff.reshape(
            1, -1).dot(imat.dot(d_ds_diff.reshape(-1, 1)))).flatten()
        for idx, distance in enumerate(dist_dev):
            if distance > std:
                self.Sigma_z[idx] = 99999

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

        innovation = np.array(list(self.observations.values())).flatten(  ##dict_values -> list -> np.array [Saimai]
        ) - np.linalg.norm(self.x_sb - current_mu, axis=1)
        n_mu = mu+K_t@innovation
        n_cov = np.dot((np.eye(self.d) - np.dot(K_t, H_t)), cov)
        n_cov = 0.5*(n_cov + np.transpose(n_cov))
        # print (n_mu, n_cov)
        n_mu = n_mu.flatten()
        return n_mu, n_cov

    def ekf_predict(self, mu, cov, dt):
        """
        Turtlebot differential drive dynamics
        """
        Ak = np.array([[1., 0., 0., -self.u[0]*np.sin(mu[3])*dt],
                       [0., 1., 0., self.u[0]*np.cos(mu[3])*dt],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]])

        Bk = np.array([[np.cos(mu[2])*dt, 0],
                      [np.sin(mu[2])*dt, 0],
                      [0., 0.],
                       [0, self.u[1]*dt]])

        # Prediction step
        n_mu = np.dot(Ak, mu.reshape(-1, 1)) + \
            np.dot(Bk, self.u.reshape(-1, 1))
        n_cov = np.dot(Ak, np.dot(cov, Ak.T)) + self.W
        return (n_mu.flatten(), n_cov)

    def inputCallback(self, sub_input=[0, 0]):
        self.u = np.array([sub_input.linear.x, sub_input.angular.z]).flatten()

    def positionCallback(self, stat_pos):
        if stat_pos.address in self.observations:  ##added to prevent error when somehow beacon 99 is not showing up in beacon_raw_distance [Saimai]
            self.stat_beacons[stat_pos.address] = [
                stat_pos.x_m, stat_pos.y_m, stat_pos.z_m]
            self.d_dim = len(self.stat_beacons)
            self.Sigma_z = np.eye(self.d_dim)
            self.x_sb = list(self.stat_beacons.values()) #made dict_values into list

    def distanceCallback(self, dist):
        self.observations[dist.address_beacon] = dist.distance_m
        self.my_address = dist.address_hedge

class newNode(Node):
    def __init__(self):
        super().__init__("something")
        self.declare_parameter("robotName", "NameSpace")
        # self.name = self.get_parameter('robotName').get_parameter_value().string_value
        self.name = self.get_namespace()
        self.name2 = ""
        if self.name=="/":
            self.name = ""
        else:
            self.name2 = self.name+"T"
        d = 4  # Working in planar domain, x,y,z, yaw theta
        self.efilter = Ekf(d)
        self.mu, self.cov = self.efilter.mu, self.efilter.cov
        self.mm_input = self.create_subscription(BeaconPositionAddressed, self.name2+"/beacons_pos_addressed", self.efilter.positionCallback,
                                            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.sub_input = self.create_subscription(Twist, self.name+"/cmd_vel", self.efilter.inputCallback,
                                             QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.dist_input = self.create_subscription(BeaconDistance, self.name2+"/beacon_raw_distance", self.efilter.distanceCallback,
                                              QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        
        self.imu_sub = self.create_subscription(Imu, self.name+"/imu", self.imuCallback, 
                                                        QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        self.pub_pos = self.create_publisher(Odometry, self.name+"/all_positions", 1000)

        print("Wait a few seconds to initialize, then press ENTER.")
        a = input()
        self.create_timer(0.1, self.timer_callback)
        self.cmd = Odometry()

    def imuCallback(self, msg: Imu):
        self.get_logger().info("Got IMU.")
        self.cmd.pose.pose.orientation.x = msg.orientation.x
        self.cmd.pose.pose.orientation.y = msg.orientation.y
        self.cmd.pose.pose.orientation.z = msg.orientation.z
        self.cmd.pose.pose.orientation.w = msg.orientation.w

    def timer_callback(self):
        T = 10  # hz
        try:
            efilter = self.efilter
            if efilter.x_sb is not None:
                print('Mean estimate', self.mu, efilter.observations)
                [self.n_mu, self.n_cov] = efilter.ekf_predict(self.mu, self.cov, 1./T)

                efilter.acceptance(self.n_mu)
                [self.mu, self.cov] = efilter.ekf_observe(self.n_mu, self.n_cov)
            
            self.cmd.pose.pose.position.x = self.mu[0]
            self.cmd.pose.pose.position.y = self.mu[1]
            self.cmd.pose.pose.position.z = self.mu[2]
            # msg.covariance = np.linalg.eig(cov)[0][0]
            # print(cov.shape)
            # msg.covariance = self.cov
            # print(self.cov.shape)
            # print(msg.covariance)

            self.pub_pos.publish(self.cmd)
        except:
            print("something is wrong, wait a while before all becons are found.")

        


def main(args=None):
    rclpy.init(args=args)
    node = newNode();
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown
    # rospy.init_node('ekf_node')
    # T = 10  # hz
    # rate = rospy.Rate(T)
    # d = 4  # Working in planar domain, x,y,z, yaw theta
    # efilter = Ekf(d)
    # mu, cov = efilter.mu, efilter.cov

    # mm_input = rospy.Subscriber(
    #     '/yosemite/beacons_pos_addressed', BeaconPositionAddressed, efilter.positionCallback)
    
    # stat_pos = rospy.wait_for_message('/beacons_pos_a', beacon_pos_a,timeout=0.5)
    # efilter.positionCallback(stat_pos)

    # sub_input = rospy.Subscriber(
    #     '/yosemite/cmd_vel', Twist, efilter.inputCallback)
    # dist_input = rospy.Subscriber(
    #     '/yosemite/beacon_raw_distance', BeaconDistance, efilter.distanceCallback)

    # pub_pos = rospy.Publisher('/all_positions', PoseWithName, queue_size=1000)
    # pub_kal = rospy.Publisher('afterKalman', PoseWithName, queue_size=1)
    # pub_name = rospy.Publisher('nametest', PoseWithName, queue_size=5)

    # while not rospy.is_shutdown():
    #     if efilter.x_sb is not None:
    #         print('Mean estimate', mu, efilter.observations)
    #         [n_mu, n_cov] = efilter.ekf_predict(mu, cov, 1./T)

            # if efilter.acceptance(n_mu):#True:#
            #     [mu, cov] = efilter.ekf_observe(n_mu, n_cov)
            #     # print ('Observed!')
            # else:
            #     [mu, cov] = [n_mu, n_cov]
            #     print ('Diffident!')

            # efilter.acceptance(n_mu)
            # [mu, cov] = efilter.ekf_observe(n_mu, n_cov)

        # pub_pos.publish(mu)
        # pub_kal.publish(mu)
        # pub_name.publish(mu)

        # rate.sleep()


if __name__ == '__main__':
    main()
