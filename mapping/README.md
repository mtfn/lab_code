# Mapping Package Setup and Usage

* [Setup](#setup)
* [Start single mapping](#start-single-mapping)
* [Plotting](#plotting)
* [Resetting](#resetting-for-new-map)
* [Debug](#debug)
* [More options](#advanced-options)

## Setup
**ssh into turtlebot**
````console
user@local:~$ ssh ubuntu@192.168.185.***
ubuntu@192.168.185.***'s password: turtlebot4
ubuntu@192.168.185.***:~$ cd ros2_ws
````
Install `scipy` and `pickle` on turtlebot if they're not inistalled.

**Download and put these on turtlebot inside /home/ubuntu/ros2_ws/src**

* marvelmind_ros2
* marvelmind_ros2_msgs
* turtlebot4_mekf
* lidar_to_global
* mapping

**Then,**

````
cd /home/ubuntu/ros2_ws
colcon build --symlink-install --packages-select marvelmind_ros2 marvelmind_ros2_msgs turtlebot4_mekf lidar_to_global mapping
source install/setup.bash
````

(omit marvelmind_ros2 marvelmind_ros2_msgs if they are already there.)

**I'm using "yosemite" in place of the namespace, change it to the namespace of the turtlebot.**

**Launch the localization package once to get angle difference from odom frame to marvelmind frame. Wait for the turtlebot to mave forward and stop, a file `/home/ubuntu/yosemiteInitYaw.pcl` will be created.** 
(If you restart the turtlebot, delete the file and generate it again.)
```
ros2 launch turtlebot4_mekf turtlebot4_mekf.launch.py namespace:=/yosemiteT
```
(It is actually just EKF now, but the filename is from previous version.)

**Then, hit `Ctrl+c` to stop it and start lidar with**

````
ros2 launch lidar_to_global lidar.launch.py namespace:=/yosemiteT
````
**Make sure the turtlebot has at least 6ft of free space in front of it. Wait for a few minutes, then stop it with `Ctrl+c`. Sampled points are saved to `/home/ubuntu/scannedPoints.txt`**

**Done, make sure the 2 files are created.**

## Start single mapping
**Having done the setup, make sure there is at least 6ft of space, then launch single robot mapping with this one line of command.**
````
ros2 launch mapping mapping.launch.py namespace:=/yosemiteT
````
A file `/home/ubuntu/fpl.pcl` should be created and will be reused in future launches.
Note: This is very slow on turtlebot, be patient with it.

## Plotting 
(First plot that pops up needs to be closed for the code to continue)
**On your computer, install the packages and do**
````
ros2 run turtlebot4_mekf plot_path
````
**Then enter `/yosemite/pose_pub` or `/yosemite/scannedPoints` to plot the position or the converted lidar data.**

**For plotting the map,**
````
ros2 run mapping plotData
````
Then follow the prompt, e.g.

````console
Path to .pcl with fpoints, Xtest,...: /home/saimai/fpl.pcl
Single robot? (1/0) 1
Full omega/cov? (1/0) 1
Mu, omega/cov in string? (1/0) 0
Number of turtlebots: 1
Reading mu, cov from: yosemite
Plot map instead of error? (1/0) 1
````

## Resetting for new map
**Delete `fpl.pcl` and `yosemiteInitYaw.pcl`, then redo second half of Setup.** (i.e. After building the packages.)

## Debug

**If Lidar is not working or topics are hidden, in the turtlebot, try**
````
sudo systemctl stop turtlebot4.service
ros2 daemon stop
sudo systemctl start turtlebot4.service
ros2 daemon start
````
**To check,**
````
ros2 topic echo /yosemite/scan
````
**and see if it prints any data.**
**Repaeat a few times if it still doesn't work.**

## Advanced options
````
ros2 launch mapping mapping.launch.py namespace:=/yosemiteT speed:=0.07 range:=6.0 fpl:=/home/ubuntu/fpl.pcl
````

### Note: Works too without the extra "T" in namespace, but from my observation it has a higher chance of working with the extra "T" added.

