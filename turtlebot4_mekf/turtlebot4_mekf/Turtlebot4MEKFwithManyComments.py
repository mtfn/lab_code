#!/usr/bin/env python3
# MIT License
import numpy as np
import sys
import rclpy
import tf2_ros
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_services_default
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped, Quaternion, TransformStamped
# from turtlebot_deployment.msg import PoseWithName
from marvelmind_ros2_msgs.msg import BeaconDistance, BeaconPositionAddressed
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from turtlebot4_mekf.util import skewSymmetric, quatToMatrix, quatMultiply, quatNorm
# from . import util
from queue import PriorityQueue

# Multiplicative EKF ROS2 Node for Marvelmind and IMU Sensor Fusion


class Turtlebot4Mekf(Node):
    """
    States: 9 or 15. 
    [0:3] attitude error quaternion to encode orientation uncertainty. [3x1 vector] dq(alpha) = (1 alpha/2).T, q = q_est dq(alpha)
    [3:6] velocity vector in body frame
    [6:9] position vector in inertial frames

    Optional
    [10:12] gyro bias
    [13:15] accelerometer bias
    """

    def __init__(self):
        # Create Node
        super().__init__('MEKF')
        
        self.name = self.get_namespace()
        
        if self.name=="/":
            self.name = "";
        
        self.estimate_bias = True
        self.isMoving = True

        # Declare params
        parameters = self.declare_parameters(  # all params should be float or int or string or array of those data types
            namespace=self.name,  # maybe should have something here
            parameters=[
                ('node_rate', 25),
                ('robot_name', 'yosemite'),
                ('beacon_type', 'marvelmind'),  # marvelmind or pozyx
                ('init_quat', [0., 0., 0., 1.]),  # x,y,z,w
                ('init_quat_cov', [0.1, 0.1, 0.1]),
                ('init_vel', [0., 0., 0.]),
                ('init_vel_cov', [0.1, 0.1, 0.1]),
                ('init_pos', [0., 0., 0.]),
                ('init_pos_cov',  [0.25, 0.25, 0.1]),
                ('gyro_cov',  [0.1, 0.1, 0.1]),
                ('gyro_bias_cov',  [0.01, 0.01, 0.01]),
                ('accel_cov',  [0.1, 0.1, 0.1]),
                ('accel_bias_cov',  [0.01, 0.01, 0.01]),
                ('uwb_obs_cov',  0.01),
                ('mahalanobis_dist', 3.),
                ('acc_gravity', [0., 0., 0.])  # 1 or 9.81
            ]
        )
        parameter_names = [parameter.name for parameter in parameters]

        [node_rate, robot_name, beacon_type,
         init_quat, init_quat_cov, init_vel,
         init_vel_cov, init_pos, init_pos_cov,
         gyro_cov, gyro_bias_cov, accel_cov,
         accel_bias_cov, uwb_obs_cov, self.mahalanobis_dist, acc_gravity] = [parameter.value for parameter in self.get_parameters(parameter_names)]

        self.rate = node_rate
        # Make priority queues to order measurements
        max_q_size = 10  # decide after testing for performance, will vary for marvelmind and pozyx
        self.q = PriorityQueue(maxsize=max_q_size)

        # Subscribers QOS similar to queue size in ROS1, default is 10
        if beacon_type == 'marvelmind':
            self.sub_marvelmind_pos = self.create_subscription(
                BeaconPositionAddressed,
                self.name+"/beacons_pos_addressed",
                self.marvelmindPosCallback,
                qos_profile_sensor_data)
            self.sub_marvelmind_dist = self.create_subscription(
                BeaconDistance,
                self.name+"/beacon_raw_distance",
                self.marvelmindDistCallback,
                10)
        elif beacon_type == 'pozyx':
            pass
            # self.sub_pozyx_pos = self.create_subscription()
            # self.sub_pozyx_dist = self.create_subscription()
        else:
            pass
            # make correct error, and log wrong beacon type or param not declared
        self.sub_imu = self.create_subscription(
            Imu,
            self.name+"/imu",
            self.imuCallback,
            qos_profile_sensor_data)
        self.sub_cmd_vel = self.create_subscription(
            Twist,
            self.name+"/cmd_vel",
            self.cmdvelCallback,
            10)

        # # Publishers
        # self.pub_pos = self.create_publisher(
        #     '/all_positions',
        #     PoseWithName,
        #     100)
        # self.pub_kal = self.create_publisher(
        #     'afterKalman',
        #     PoseWithName,
        #     1)
        # self.pub_name = self.create_publisher(
        #     'nametest',
        #     PoseWithName,
        #     5)

        self.pub_odom = self.create_publisher(
            Odometry,
            self.name+"/odometry/filtered",
            qos_profile=qos_profile_services_default
        )

        # Initialization
        if self.estimate_bias:
            self.num_states = 15
        else:
            self.num_states = 9

        # stat beacon position map for quick location based on beacon address
        self.stat_beacons = {}
        # nominal estimate
        self.q_est = init_quat  # geometry_msgs Quaternion (x,y,z,w)
        self.vel_est = np.array(init_vel, dtype=float)
        self.pos_est = np.array(init_pos, dtype=float)
        self.ang_vel_est = np.array([0., 0., 0.])

        # Convert params to numpy arrays that are covariance matrices
        init_quat_cov = np.array(init_quat_cov)*np.identity(3, dtype=float)
        init_vel = np.array(init_vel)
        init_vel_cov = np.array(init_vel_cov)*np.identity(3, dtype=float)
        init_pos = np.array(init_pos)
        init_pos_cov = np.array(init_pos_cov)*np.identity(3, dtype=float)
        self.gyro_cov = np.array(gyro_cov)*np.identity(3, dtype=float)
        self.gyro_bias_cov = np.array(
            gyro_bias_cov)*np.identity(3, dtype=float)
        self.accel_cov = np.array(accel_cov)*np.identity(3, dtype=float)
        self.accel_bias_cov = np.array(
            accel_bias_cov)*np.identity(3, dtype=float)
        uwb_obs_cov = np.array(uwb_obs_cov)
        self.acc_gravity = np.array(acc_gravity, dtype=float)

        # Initialize error state estimate at 0
        self.estimate = np.zeros(shape=(self.num_states, ), dtype=float)
        self.estimate_covariance = np.zeros(
            shape=(self.num_states, self.num_states), dtype=float)
        self.estimate_covariance[0:3, 0:3] = init_quat_cov  # 3x3 rad
        self.estimate_covariance[3:6, 3:6] = init_vel_cov  # 3x3
        self.estimate_covariance[6:9, 6:9] = init_pos_cov  # 3x3

        if self.estimate_bias:
            # part of estimate
            self.gyro_bias = np.array([0.0, 0.0, 0.0])
            self.accelerometer_bias = np.array([0.0, 0.0, 0.0])
            # part of estimate_cov
            self.estimate_covariance[9:12, 9:12] = self.gyro_bias_cov  # 3x3
            self.estimate_covariance[12:15, 12:15] = self.accel_bias_cov  # 3x3

        # Initialize observation covariance for beacons
        self.observation_covariance = uwb_obs_cov
        # Initialize matrices
        # Orientation kinematics
        self.F = np.zeros(
            shape=(self.num_states, self.num_states), dtype=float)
        self.F[6:9, 3:6] = np.identity(3, dtype=float)
        if self.estimate_bias:
            self.F[0:3, 9:12] = -np.identity(3, dtype=float)
        
        print("Initialized, starting filtering")
        self.time_counter = 0
        self.init_time = self.get_clock().now() 
        # Create loop for kalman filtering
        self.timer = self.create_timer(1./self.rate, self.timer_callback)

    def timer_callback(self):
        self.mekfT = self.get_clock().now()
        # self.pubOdom()
        print("queue length",self.q.qsize())
        while not self.q.empty():
            # check if times are equal in q
            measurement = self.q.get()
            if measurement[1] == 'imu':
                # print("IMU Prediction")
                if self.isMoving:  # check if there are velocity inputs
                    (t, _, gyro_meas, acc_meas) = measurement
                    self.mekfPredict(np.array([gyro_meas.x, gyro_meas.y, gyro_meas.z]), np.array(
                        [acc_meas.x, acc_meas.y, acc_meas.z]), t)
                    self.isMoving = True 
            elif measurement[1] == 'dist':
                print("Beacon Update")
                sys.stdout.flush()
                # if self.distAcceptance(measurement[2], measurement[3]):
                (t, _, dist_meas, pos_st) = measurement
                self.mekfUpdate(dist_meas, pos_st, t)
            # elif np.floor(((self.mekfT-self.init_time).to_msg().nanosec*1e-9)/(1./self.rate)) < self.time_counter: 
              #  self.pubOdom()
               # self.time_counter += 1
            else:
                # print("queue is messed up")  # log
                pass
        # print("No Measurements")
        # Odometry publisher
        self.pubOdom()

    def mekfPredict(self, gyro_meas, acc_meas, t):
       # print("Prediction Step")
        self.mekfT = self.get_clock().now()
        time_delta = (self.mekfT-t).to_msg().nanosec*1e-9
        if time_delta < 0.:
            # log error
            # print("Messed up timing")
            pass
        self.mekfT = t

        # check to make sure bias is not too large.
        # gyro_meas = gyro_meas - self.gyro_bias
        # acc_meas = acc_meas - self.accelerometer_bias

        # Estimate nominal state through integration using gyro and acc measurements
        # Can try averaged inputs if necessary from (94) in [4]
        self.q_est += time_delta*0.5 * \
            quatMultiply(self.q_est, [gyro_meas[0],
                         gyro_meas[1], gyro_meas[2], 0])
        self.q_est = quatNorm(self.q_est)
        self.vel_est += (np.dot(quatToMatrix(self.q_est),
                         acc_meas) + self.acc_gravity)*time_delta
        self.pos_est += self.vel_est*time_delta
        self.ang_vel_est = np.array([gyro_meas[0], gyro_meas[1], gyro_meas[2]])

        # Error kinematics to form state transition matrix
        self.F[0:3, 0:3] = -skewSymmetric(gyro_meas)
        self.F[3:6, 0:3] = np.dot(-quatToMatrix(self.q_est),
                                  skewSymmetric(acc_meas))
        if self.estimate_bias:
            self.F[3:6, 12:15] = -quatToMatrix(self.q_est)
        state_transition_matrix = np.identity(
            self.num_states, dtype=float) + self.F*time_delta + 0.5*time_delta**2*np.dot(self.F, self.F)

        # Update with error state a priori covariance
        self.estimate_covariance = np.dot(np.dot(state_transition_matrix, self.estimate_covariance),
                                          state_transition_matrix.T) + self.process_covariance(time_delta)
        # print(self.estimate_covariance.shape)

    def mekfUpdate(self, dist_meas, pos_st, t):
        # print("Measurement Step")
        self.mekfT = self.get_clock().now()
        if self.mekfT > t:
            # log error
            pass
        self.mekfT = t
        H = np.zeros(shape=(1, self.num_states), dtype=float)
        dist = np.linalg.norm(self.pos_est - pos_st)
        # print("dist",dist)
        H[0, 6:9] = (self.pos_est-pos_st)/dist

        PH_T = np.dot(self.estimate_covariance, H.T)
        # print("PH_T",PH_T,"H_T",H.T)
        innovation_cov = H.dot(PH_T) + self.observation_covariance
        # print("innovation_cov",innovation_cov.shape,innovation_cov,"H.dot(PH_T",H.dot(PH_T))
        # TODO DEBUG HERE
        # print("observation cov",self.observation_covariance.size,self.observation_covariance,"H.dot(PH_T)",H.dot(PH_T).shape )
        # if not self.distAcceptance(dist_meas, pos_st, innovation_cov):
        #     return
        # Form Kalman gain
        # print("inv innov cov",np.linalg.inv(innovation_cov))
        K = np.dot(PH_T, np.linalg.inv(innovation_cov))
        # print("K",K)
        # Update with a posteriori covariance, make sure to make it symmetric for stability
        self.estimate_covariance = (np.identity(
            self.num_states, dtype=float) - np.dot(K, H)).dot(self.estimate_covariance)
        self.estimate_covariance = 0.5 * \
            (self.estimate_covariance + self.estimate_covariance.T)

        # measurement residual
        measurement_residual = dist_meas - \
            np.linalg.norm(self.pos_est - pos_st)
       ## print('meas residual', measurement_residual, 'pos_est', self.pos_est,
             # 'stat pos', pos_st, 'est dist', np.linalg.norm(self.pos_est-pos_st))
        # calculate a posteriori update
        aposteriori_state = np.dot(
            K, measurement_residual.T).flatten()  # (15,)
        # print("aposteriori state", aposteriori_state,"measurement residualt", measurement_residual)

        # Fold filtered error state back into full state estimates
        self.q_est = quatMultiply(self.q_est, [
                                  0.5*aposteriori_state[0], 0.5*aposteriori_state[1], 0.5*aposteriori_state[2], 1])
        self.q_est = quatNorm(self.q_est)
        self.vel_est += aposteriori_state[3:6]
        self.pos_est += aposteriori_state[6:9]
        if self.estimate_bias:
            self.gyro_bias += aposteriori_state[9:12]
            self.accelerometer_bias += aposteriori_state[12:15]
           # print(self.gyro_bias, self.accelerometer_bias)

        # print(self.pos_est, self.vel_est, self.q_est)

    def process_covariance(self, time_delta):
        # Is the From (97) in [4]
        Q = np.zeros(shape=(self.num_states, self.num_states), dtype=float)
        Q[0:3, 0:3] = self.gyro_cov*time_delta + \
            self.gyro_bias_cov*(time_delta**3)/3.0
        Q[0:3, 9:12] = -self.gyro_bias_cov*(time_delta**2)/2.0
        Q[3:6, 3:6] = self.accel_cov*time_delta + \
            self.accel_bias_cov*(time_delta**3)/3.0
        Q[3:6, 6:9] = self.accel_bias_cov * \
            (time_delta**4)/8.0 + self.accel_cov*(time_delta**2)/2.0
        Q[3:6, 12:15] = -self.accel_bias_cov*(time_delta**2)/2.0
        Q[6:9, 3:6] = self.accel_cov * \
            (time_delta**2)/2.0 + self.accel_bias_cov*(time_delta**4)/8.0
        Q[6:9, 6:9] = self.accel_cov * \
            (time_delta**3)/3.0 + self.accel_bias_cov*(time_delta**5)/20.0
        Q[6:9, 12:15] = -self.accel_bias_cov*(time_delta**3)/6.0
        Q[9:12, 0:3] = -self.gyro_bias_cov*(time_delta**2)/2.0
        Q[9:12, 9:12] = self.gyro_bias_cov*time_delta
        Q[12:15, 3:6] = -self.accel_bias_cov*(time_delta**2)/2.0
        Q[12:15, 6:9] = -self.accel_bias_cov*(time_delta**3)/6.0
        Q[12:15, 12:15] = self.accel_bias_cov*time_delta
        return Q

    def distAcceptance(self, dist_meas, pos_st, innovation_cov):
        """
        Assume current robot position is given by the normal distribution N(mu, Sigma). The Mahalanobis distance(m) is used for acceptance, which characterizes the distance a measurement is from the estimate distribution. If the measurement is deemed outside a m of std, it is rejected. Acceptance is for one individual static beacon measurement.
        """
        measurement_residual = dist_meas-np.linalg.norm(self.pos_est-pos_st)
        inv_innovation_cov = np.linalg.inv(innovation_cov)  # current pos cov
        m_distance = np.sqrt(
            np.dot(np.dot(measurement_residual, inv_innovation_cov), measurement_residual))
        # print(m_distance[0, 0], inv_innovation_cov, measurement_residual)
        if m_distance[0, 0] < self.mahalanobis_dist:
            return True
        else:
            # Measurement above std so rejected
            # print("Measurement Rejected, std=%d", m_distance)
            return False

    def pubOdom(self):
        odom_broadcaster = tf2_ros.TransformBroadcaster(self)
        t = TransformStamped()
        current_time = self.mekfT.to_msg()
        t.header.stamp = current_time
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = self.pos_est[0]
        t.transform.translation.y = self.pos_est[1]
        t.transform.translation.z = self.pos_est[2]
        t.transform.rotation.x = self.q_est[0]
        t.transform.rotation.y = self.q_est[1]
        t.transform.rotation.z = self.q_est[2]
        t.transform.rotation.w = self.q_est[3]
        # publish base
        odom_broadcaster.sendTransform(t)

        odom_msg = Odometry()
        odom_msg.header.frame_id = "odom"
        odom_msg.header.stamp = current_time

        # set the position
        odom_msg.pose.pose.position.x = self.pos_est[0]
        odom_msg.pose.pose.position.y = self.pos_est[1]
        odom_msg.pose.pose.position.z = self.pos_est[2]
        odom_msg.pose.pose.orientation.x = self.q_est[0]
        odom_msg.pose.pose.orientation.y = self.q_est[1]
        odom_msg.pose.pose.orientation.z = self.q_est[2]
        odom_msg.pose.pose.orientation.w = self.q_est[3]

        # set the velocity
        odom_msg.child_frame_id = "base_link"
        odom_msg.twist.twist.linear.x = self.vel_est[0]
        odom_msg.twist.twist.linear.y = self.vel_est[1]
        odom_msg.twist.twist.linear.z = self.vel_est[2]
        # 0 if no imu data
        odom_msg.twist.twist.angular.x = self.ang_vel_est[0]
        odom_msg.twist.twist.angular.y = self.ang_vel_est[1]
        odom_msg.twist.twist.angular.z = self.ang_vel_est[2]
        # TODO add covariances
        self.pub_odom.publish(odom_msg)

    def marvelmindPosCallback(self, stat_pos):
        # Static beacon positions
        self.stat_beacons[stat_pos.address] = [
            stat_pos.x_m, stat_pos.y_m, stat_pos.z_m]
        # self.num_beacons = len(self.stat_beacons)
        # self.Sigma_z = np.eye(self.num_beacons)
        # self.x_sb = self.num_beacons.values

    def marvelmindDistCallback(self, dist):
        # https://github.com/mikeferguson/ros2_cookbook/blob/main/rclpy/time.md
        t = self.get_clock().now()
        if self.stat_beacons:
            try:
                self.q.put((t, 'dist', dist.distance_m, np.array(
                    self.stat_beacons[dist.address_beacon])))

                # self.prev_dist_t = t
            except KeyError:
                print('beacon address KeyError when getting position')
                print('stat beacon, address, dist',self.stat_beacons, dist.address_beacon, dist.distance_m)

    def imuCallback(self, imu_msg):
        t = self.get_clock().now()
        self.q.put((t, 'imu', imu_msg.angular_velocity,
                   imu_msg.linear_acceleration))

    def cmdvelCallback(self, cmd_vel_msg):
        self.isMoving = True 
        velx = cmd_vel_msg.linear.x
        velangz = cmd_vel_msg.angular.z
        tol = 1e-1
        if (abs(velx)-tol < 0) or (abs(velangz)-tol < 0):
            self.isMoving = False
            
    # Implement later
    # def pozyxPosCallback(self, stat_pos):
    #     # Static UWB beacon positions
    #     self.stat_beacons[stat_pos.address] = [
    #         stat_pos.x_m, stat_pos.y_m, stat_pos.z_m]
    #     self.num_beacons = len(self.stat_beacons)
    #     self.Sigma_z = np.eye(self.num_beacons)
    #     self.x_sb = self.num_beacons.values()

    # def pozyxDistCallback(self, device_range):
    #     self.stadft_beacons[device_range.address] = [device_range.distance, device_range.timestamp]


def main(args=None):
    rclpy.init(args=args)
    turtlebot4_mekf = Turtlebot4Mekf()
    # rclpy.spin(turtlebot4_mekf)
    try:
        rclpy.spin(turtlebot4_mekf)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        rclpy.try_shutdown()
        turtlebot4_mekf.destroy_node()
    # turtlebot4_mekf.destroy_node()
    # rclpy.shutdown()


if __name__ == "__main__":
    main()
