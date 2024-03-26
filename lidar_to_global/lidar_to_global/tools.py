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
    
class followPath:
    def __init__(self, velOut, speed, pos, yaw):
        self.velOut = velOut
        self.maxSpeed = speed
        self.move = False
        self.dt = 0.5
        # self.lincon = PID(Kp = 0.08, Ki = 0.01, Kd = 0.005, max = self.maxSpeed, min = 0, dt = self.dt)
        self.angcon = PID(Kp = 0.3, Ki = 0.05, Kd = 0.1, max = np.pi, min = -np.pi, dt = self.dt)
        self.pos = pos
        self.yaw = yaw
    def stop(self):
        self.move = False
    def go(self, loc: np.array):
        self.move = True
        cmd1 = Twist()
        distConst = 0.1
        p1a = self.getAngle(self.pos,self.center)
        closestPoint = self.center-np.array([self.radius*np.cos(p1a),self.radius*np.sin(p1a)])
        distanceToClosest = self.getDistance(self.pos,closestPoint)
        perpenAng = self.getAngle(self.pos,closestPoint)

        if distanceToClosest<distConst:
            tempTargetOnTangentLine = closestPoint+np.array([0.15*np.cos(p1a-np.pi/2),0.15*np.sin(p1a-np.pi/2)])
            direction = self.getAngle(self.pos,tempTargetOnTangentLine)
            change = self.repairAngValue(perpenAng - direction)
            change = 0.15*change/distConst*abs(distanceToClosest)

            newRefAng = direction + change
            angleErr = newRefAng-self.yaw
        else:
            angleErr = perpenAng - self.yaw
        speed  = self.maxSpeed
    
        cmd1.linear.x = speed
        cmd1.angular.z = self.angcon.uVal(self.repairAngValue(angleErr))

        self.velOut.publish(cmd1)

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

    