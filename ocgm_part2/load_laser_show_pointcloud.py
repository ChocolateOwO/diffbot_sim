import time
import matplotlib.pyplot as plt
import numpy as np
# import csv
from matplotlib.animation import FuncAnimation

data = np.loadtxt('data.csv', delimiter=',')

X = data[:,0]
Y = data[:,1]
TH = data[:,2]

num_ray = int((data.shape[1]-3)/2)

shape = (data.shape[0],num_ray) 
sensor_length = np.empty(shape)
sensor_angle = np.empty(shape)

for i in range(num_ray):
    sensor_length[:,i] = data[:,3+2*i]
    sensor_angle[:,i] = data[:,4+2*i]

# print(X.shape)
# print(Y.shape)
# print(TH.shape)
print(sensor_length.shape)
# print(sensor_angle.shape)

fig, ax = plt.subplots()
plt.gca().set_aspect('equal', adjustable='box')
# plt.axis('equal')
# Gazebo scene is 30 x 30 m (x, y in -15..15); axes are not inverted (unlike the CoppeliaSim example)
plt.xlim(-16, 16)
plt.ylim(-16, 16)
ax.set_xlim(-16, 16)
ax.set_ylim(-16, 16)
plt.draw()

x = np.random.rand(5)  # Random initial x coordinates
y = np.random.rand(5)  # Random initial y coordinates
sc = ax.scatter(x, y, color='blue', s=0.1)  # Initial scatter plot with 

global frame_num
frame_num = 0

def update(frame):
    global frame_num
    print(frame)

    Xall = np.empty((0,1))
    Yall = np.empty((0,1))
    for i in range(num_ray):
        # l_xnew, l_ynew = Transform_laser2pc(X[frame_num],Y[frame_num],TH[frame_num], sensor_length[frame_num,i], sensor_angle[frame_num,i])
        l_xnew, l_ynew = Transform_laser2pc(X[frame],Y[frame],TH[frame], sensor_length[frame,i], sensor_angle[frame,i])

        # Xall = np.vstack((Xall,l_xnew))
        Xall = np.append(Xall,l_xnew)
        Yall = np.append(Yall,l_ynew)
    # print(l_xnew)

    # print(Xall.shape)
    # print(Yall.shape)

    sc.set_offsets(np.column_stack((Xall, Yall))) 

    frame_num = frame_num + 1
    return sc,

def Transform_laser2pc(X,Y,th,le,an):
    x = le*np.cos(an)
    y = le*np.sin(an)
    xnew = X + x*np.cos(th) - y*np.sin(th)
    ynew = Y + x*np.sin(th) + y*np.cos(th)
    
    return xnew, ynew

# try:
#     ani = FuncAnimation(fig, update, frames=100, interval=100, blit=True)

# except:
#     print("something wrong")
ani = FuncAnimation(fig, update, frames=data.shape[0], interval=100, blit=False, repeat=False)
plt.show()

# for frame_num in range(data.shape[0]):
#     Xall = np.empty((0,1))
#     Yall = np.empty((0,1))
#     for i in range(num_ray):
#         l_xnew, l_ynew = Transform_laser2pc(X[frame_num],Y[frame_num],TH[frame_num], sensor_length[frame_num,i], sensor_angle[frame_num,i])

#         # Xall = np.vstack((Xall,l_xnew))
#         Xall = np.append(Xall,l_xnew)
#         Yall = np.append(Yall,l_ynew)
#     # print(l_xnew)

#     # print(Xall.shape)
#     # print(Yall.shape)

#     sc.set_offsets(np.column_stack((Xall, Yall))) 
