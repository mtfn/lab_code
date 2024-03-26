#!/usr/bin/env python
# MIT License
import numpy as np
# Warning. This Quaternion is initialized as [x,y,z,w] instead of the hamilton [w,x,y,z]

def skewSymmetric(v):
    "Return skew symmetric matrix of a 3 vector"
    return np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]])
                

def quatToMatrix(q):
    "Direction Cosine matrix from inertial to body framme"
    return 2.0*np.outer(q[0:3], q[0:3]) \
      + np.identity(3)*(q[3]**2 - np.dot(q[0:3],q[0:3])) \
      + 2*q[3]*skewSymmetric(q[0:3])


def quatInverse(q):
    "Calculate reciprocal of Quaternion"
    norm = np.sqrt(q[0]**2+q[1]**2+q[2]**2+q[3]**2)
    return [-q[0]/norm, -q[1]/norm, -q[2]/norm, q[3]/norm]


def quatNorm(q):
    "Normalize Quaternion to versor/unit Quaternion"
    norm = np.sqrt(q[0]**2+q[1]**2+q[2]**2+q[3]**2)
    return [q[0]/norm, q[1]/norm, q[2]/norm, q[3]/norm]

def quatMultiply(q0,q1):
    """
    Multiplies two Quaternions.
 
    Input
    :param q0: A 4 element array containing the first Quaternion (q11,q21,q31,q01) 
    :param q1: A 4 element array containing the second Quaternion (q12,q22,q32,q02) 
 
    Output
    :return: A 4 element array containing the final Quaternion (q13,q23,q33,q03) 
 
    """
    # Extract the values from q0
    x0 = q0[0]
    y0 = q0[1]
    z0 = q0[2]
    w0 = q0[3]
     
    # Extract the values from q1
    
    x1 = q1[0]
    y1 = q1[1]
    z1 = q1[2]
    w1 = q1[3]
     
    # Compute the product of the two Quaternions, term by term
    q0q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    q0q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    q0q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    q0q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
     
    # Create a 4 element array containing the final Quaternion
    final_Quaternion = np.array([q0q1_x, q0q1_y, q0q1_z, q0q1_w])

    return final_Quaternion


def quatToEuler(q):
    """
    Converts from Quaternion to 321 Euler angles as roll, pitch, yaw
    """
    euler = []
    # roll (x-axis rotation)
    x = np.atan2(2*(q[3]*q[0] + q[1]*q[2]), 1 - 2*(q[0]*q[0] + q[1]*q[1]))
    euler.append(x)
    # pitch (y-axis rotation)
    x = np.asin(2*(q[3]*q[1] - q[2]*q[0]))
    euler.append(x)
    # yaw (z-axis rotation)
    x = np.atan2(2*(q[3]*q[2] + q[0]*q[1]), 1 - 2*(q[1]*q[1] + q[2]*q[2]))
    euler.append(x)

    return euler


def quatListToEulerArrays(qs):
    euler = np.ndarray(shape=(3, len(qs)), dtype=float)

    for (i, q) in enumerate(qs):
        e = quatToEuler(q)
        euler[0, i] = e[0]
        euler[1, i] = e[1]
        euler[2, i] = e[2]

    return euler


def eulerError(estimate, truth):
    return np.minimum(np.minimum(np.abs(estimate - truth), np.abs(2*np.pi + estimate - truth)), np.abs(-2*np.pi + estimate - truth))


def eulerArraysToErrorArrays(estimate, truth):
    errors = []
    for i in range(3):
        errors.append(eulerError(estimate[i], truth[i]))
    return errors


def quatListToErrorArrays(estimate, truth):
    return eulerArraysToErrorArrays(quatListToEulerArrays(estimate), quatListToEulerArrays(truth))


def rmse_euler(estimate, truth):
    def rmse(vec1, vec2):
        return np.sqrt(np.mean((vec1 - vec2)**2))

    return[rmse(estimate[0], truth[0]),
           rmse(estimate[1], truth[1]),
           rmse(estimate[2], truth[2])]
