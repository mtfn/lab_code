import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rclpy.qos import ReliabilityPolicy, QoSProfile

class PID:
    def __init__(self, Kp, Ki, Kd, max, min, dt):
        self.dt = dt
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.lastErr = 1000
        self.max = max
        self.min = min
    def uVal(self,err):
        Kp = self.Kp
        Ki = self.Ki
        Kd = self.Kd
        pTerm = Kp*err
        self.integral += err*self.dt
        iTerm = Ki*self.integral
        dTerm=0
        if self.dt>0:
            dTerm = Kd*(err-self.lastErr)/self.dt
        self.lastErr = err
        output =  pTerm+iTerm+dTerm
        if output>self.max:
            output = self.max
        elif output<self.min:
            output = self.min
        return output