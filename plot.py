import matplotlib.pyplot as plt
import numpy as np

file = open("scannedPoints.txt")
# lns = file.readlines()
plt.axis([-3,6,-6,3])
for b in range(6000):
    ln = file.readline()
    a = ln.split(",")
    x = float(a[0])
    y = float(a[1])
    lb = int(a[3])
    if lb == 1:
        plt.scatter(x,y,color="yellow")
    else:
        plt.scatter(x,y,color="green",s=6)
    
plt.show()