#!/usr/bin/env python3
import numpy as np
from rclpy.node import Node
import rclpy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped, PoseWithCovariance
# from turtlebot_deployment.msg import PoseWithName
from marvelmind_ros2_msgs.msg import BeaconDistance, BeaconPositionAddressed ##beacon_distance, beacon_pos_a
from rclpy.qos import ReliabilityPolicy, QoSProfile
# Depends on turtlebot_deployment package msg PoseWithName.
# string name
# geometry_msgs/Pose pose


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
        self.observations = {}
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
        d_ds_diff = list(self.observations.values())-np.linalg.norm(current_mu-self.x_sb)
        imat = np.linalg.inv(self.Sigma_z)
        prob = np.exp(-gamma*d_ds_diff.reshape(1, -1).dot(imat.dot(d_ds_diff.reshape(-1, 1)))).flatten()
        return np.random.binomial(2, prob, 1)

    """
    Mahalanobis distance for acceptance.
    Implement acceptance for individual beacons.
    """
    def measurementQueue(self):
        # TODO https://answers.ros.org/question/280831/specifics-of-sensor-fusion-algorithm-of-robot_localization-package/
        pass

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
        d_ds_diff = list(self.observations.values())-np.linalg.norm(current_mu-self.x_sb)  #[Saimai]
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
        H_t = (current_mu - self.x_sb).T # / np.linalg.norm(self.x_sb -
               #current_mu, axis=1)).T  # d_dimx3 matrix
        H_t = np.hstack((H_t, np.zeros((self.d_dim, 1))))
        K_t = (np.linalg.solve(np.dot(H_t, (np.dot(cov, H_t.T))) +
               self.Sigma_z, np.dot(H_t, cov))).T

        innovation = np.array(list(self.observations.values())).flatten(       #[Saimai]
        ) - np.linalg.norm(self.x_sb - current_mu, axis=1)
        n_mu = mu+K_t@innovation
        n_cov = np.dot((np.eye(self.d) - np.dot(K_t, H_t)), cov)
        n_cov = 0.5*(n_cov + n_cov.T)
        # print (n_mu, n_cov)
        n_mu = n_mu.flatten()
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
                       [0, dt]])

        # Prediction step
        n_mu = np.dot(Fk, mu.reshape(-1, 1)) + \
            np.dot(Bk, self.u.reshape(-1, 1))
        n_cov = np.dot(Fk, np.dot(cov, Fk.T)) + self.W
        return (n_mu.flatten(), n_cov)

    def inputCallback(self, sub_input=[0, 0]):
        self.u = np.array([sub_input.linear.x, sub_input.angular.z]).flatten()

    def positionCallback(self, stat_pos):
        if stat_pos.address in self.observations:  ##added to prevent error when somehow beacon 99 is not showing up in beacon_raw_distance [Saimai]
            self.stat_beacons[stat_pos.address] = [
                stat_pos.x_m, stat_pos.y_m, stat_pos.z_m]
            self.d_dim = len(self.stat_beacons)
            self.Sigma_z = np.eye(self.d_dim)
            self.x_sb = list(self.stat_beacons.values())   #[Saimai]

    def distanceCallback(self, dist):
        self.observations[dist.address_beacon] = dist.distance_m
        self.my_address = dist.address_hedge


class theNode(Node):
    def __init__(self):
        super().__init__('ekf_node')
        self.name = self.get_namespace()
        self.f = 10 #Hz
        if self.name=="/":
            self.name = ""
        self.d = 4  # Working in planar domain, x,y,z, yaw theta
        self.efilter = Ekf(self.d)
        self.mu, self.cov = self.efilter.mu, self.efilter.cov

        self.mm_input = self.create_subscription(BeaconPositionAddressed, self.name+"/beacons_pos_addressed", self.efilter.positionCallback,
                                            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.sub_input = self.create_subscription(Twist, self.name+"/cmd_vel", self.efilter.inputCallback,
                                             QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.dist_input = self.create_subscription(BeaconDistance, self.name+"/beacon_raw_distance", self.efilter.distanceCallback,
                                              QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        
        self.pub_pos = self.create_publisher(PoseWithCovariance, self.name+"/all_positions", 1000)
        self.pub_kal = self.create_publisher(PoseWithCovariance, self.name+"/afterKalman", 1)
        self.pub_name = self.create_publisher(PoseWithCovariance, self.name+"/nametest", 5)

        print("Wait a few seconds to initialize, then press ENTER.")
        a = input()
        self.create_timer(1/self.f, self.timer_callback)

    def timer_callback(self):
        # try:
        if self.efilter.x_sb is not None:
            print('Mean estimate', self.mu, self.efilter.observations)
            [self.n_mu, self.n_cov] = self.efilter.ekf_predict(self.mu, self.cov, 1./self.f)

            self.efilter.acceptance(self.n_mu)
            [self.mu, self.cov] = self.efilter.ekf_observe(self.n_mu, self.n_cov)
        # except:
        #     print("Something is wrong.")
            


def main(args=None):
    rclpy.init(args=args)
    node = theNode();
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
    #     '/beacons_pos_a', beacon_pos_a, efilter.positionCallback)
    # # stat_pos = rospy.wait_for_message('/beacons_pos_a', beacon_pos_a,timeout=0.5)
    # # efilter.positionCallback(stat_pos)
    # sub_input = rospy.Subscriber(
    #     'mobile_base/commands/velocity', Twist, efilter.inputCallback)
    # dist_input = rospy.Subscriber(
    #     '/beacon_raw_distance', beacon_distance, efilter.distanceCallback)

    # pub_pos = rospy.Publisher('/all_positions', PoseWithName, queue_size=1000)
    # pub_kal = rospy.Publisher('afterKalman', PoseWithName, queue_size=1)
    # pub_name = rospy.Publisher('nametest', PoseWithName, queue_size=5)

    # while not rospy.is_shutdown():
    #     if efilter.x_sb is not None:
    #         print('Mean estimate', mu, efilter.observations)
    #         [n_mu, n_cov] = efilter.ekf_predict(mu, cov, 1./T)

    #         # if efilter.acceptance(n_mu):#True:#
    #         #     [mu, cov] = efilter.ekf_observe(n_mu, n_cov)
    #         #     # print ('Observed!')
    #         # else:
    #         #     [mu, cov] = [n_mu, n_cov]
    #         #     print ('Diffident!')

    #         efilter.acceptance(n_mu)
    #         [mu, cov] = efilter.ekf_observe(n_mu, n_cov)

    #     #  pub_pos.publish(mu)
    #     # pub_kal.publish(mu)
    #     # pub_name.publish(mu)

    #     rate.sleep()


if __name__ == '__main__':
    main()
