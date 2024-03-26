import math
import numpy as np
import random
from sensor_msgs.msg import LaserScan

"""
This script contains functions that create a plot of mapped obstalce points given LiDAR scan data 
"""
def get_coords(lidar_data: LaserScan, dotsCnt):
    """
    Obtains x and y coordinates from LiDAR scan data by multiplying measured distance by measure angle increment of laser that detected the obstacles
    """
    coords = [] # intialize coordinates values
    n = 0 # initialize 0 for start angle

    # iterate through each individual measurement of single LiDAR scan and append to coordinate list
    for measurement in lidar_data.ranges:
        ang = lidar_data.angle_min + np.multiply(n, lidar_data.angle_increment) # increase angle to account for rotating LiDAR 
        
        n+=1 # increase angle increment

        if np.isinf(measurement) or measurement < 0.5:
            continue
        
        # compute x and y coordinate points (note: 2D LiDAR, z = 0)
        tem = [a+random.random() for a in range(dotsCnt)]
        tem.append(dotsCnt)
        dis = np.array([measurement*a/float(dotsCnt) for a in tem])
        x = np.cos(ang) * dis
        y = np.sin(ang) * dis
        z = -0.17
        for i in range(dotsCnt+1):
            coords.append([x[i], y[i], 1]) # append data to a single list

    return coords

def get_coords2(lidar_data: LaserScan, dotsCnt):
    """
    Obtains x and y coordinates from LiDAR scan data by multiplying measured distance by measure angle increment of laser that detected the obstacles
    """
    coords = [] # intialize coordinates values
    n = 0 # initialize 0 for start angle

    # iterate through each individual measurement of single LiDAR scan and append to coordinate list
    ei = [i for i,a in enumerate(lidar_data.ranges) if not(np.isinf(a) or a<0.5)]
    n = len(ei)
    angs = lidar_data.angle_min+ei*lidar_data.angle_increment
    ranges = np.array(lidar_data.ranges)
    coords = np.array([[dis*np.cos(ang), dis*np.sin(ang), 1] for dis, ang in zip(ranges,angs)])
    # for measurement in lidar_data.ranges:
    #     ang = lidar_data.angle_min + np.multiply(n, lidar_data.angle_increment) # increase angle to account for rotating LiDAR 
        
    #     n+=1 # increase angle increment

    #     if np.isinf(measurement) or measurement < 0.5:
    #         continue
        
    #     # compute x and y coordinate points (note: 2D LiDAR, z = 0)
    #     tem = [a+random.random() for a in range(dotsCnt)]
    #     tem.append(dotsCnt)
    #     dis = np.array([measurement*a/float(dotsCnt) for a in tem])
    #     x = np.cos(ang) * dis
    #     y = np.sin(ang) * dis
    #     z = -0.17
    #     for i in range(dotsCnt+1):
    #         coords.append([x[i], y[i], 1]) # append data to a single list

    return coords
