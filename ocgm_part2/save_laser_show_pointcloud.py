import time
import coppeliasim_zmqremoteapi_client as zmqRemoteApi
import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.animation import FuncAnimation

# Create a client instance
client = zmqRemoteApi.RemoteAPIClient()

# Get a handle to the simulation object
sim = client.getObject('sim')

# Start the simulation
sim.startSimulation()

ground_handle = sim.getObject('/DefaultFloor')
sensorbase_handle = sim.getObject('/ePuck')
sensorhandle1=sim.getObject('/ePuck/fastHokuyo/sensor1')
sensorhandle2=sim.getObject('/ePuck/fastHokuyo/sensor2')

R_sb_s1 = sim.getObjectMatrix(sensorhandle1,sensorbase_handle)
R_sb_s2 = sim.getObjectMatrix(sensorhandle2,sensorbase_handle)

fig, ax = plt.subplots()
plt.gca().set_aspect('equal', adjustable='box')
# plt.axis('equal')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.draw()

x = np.random.rand(5)  # Random initial x coordinates
y = np.random.rand(5)  # Random initial y coordinates
sc = ax.scatter(x, y, color='blue', s=0.1)  # Initial scatter plot with 

f = open('data.csv', 'w')
f = open('data.csv', 'a', newline='')
writer = csv.writer(f)

# Initialization function: plot the background of each frame
# def init():
#     sc.scatter(x, np.sin(x), color='blue')  # Create a masked array
#     return sc


def update(frame):
    if frame >= 800:
        ani.event_source.stop()
        
    print(frame)

    data = np.array([])

    position = sim.getObjectPosition(sensorbase_handle, ground_handle)
    # print(f"position: {position}")
    orientation = sim.getObjectOrientation(sensorbase_handle, ground_handle)
    # print(f"orientation: {orientation}")

    res1, dist1, point1 = sim.readVisionSensor(sensorhandle1)
    res2, dist2, point2 = sim.readVisionSensor(sensorhandle2)
    
    data = np.append(data, [position[0],position[1],orientation[2]])

    l_x, l_y, length_angle = get_laser_data(point1, point2)
    # print(length_angle)

    data = np.append(data, length_angle)
    writer.writerow(data)
    
    l_xnew, l_ynew = Transofrm_point(position[0],position[1],orientation[2], l_x, l_y)

    sc.set_offsets(np.column_stack((l_xnew, l_ynew))) 
    return sc,

def get_laser_data(point1, point2):
    skip = 4
    data = np.array(point1)
    x1 = np.array(data[2::skip])
    y1 = np.array(data[3::skip])
    z1 = np.array(data[4::skip])
    l1 = np.array(data[5::skip])

    data2 = np.array(point2)
    x2 = np.array(data2[2::skip])
    y2 = np.array(data2[3::skip])
    z2 = np.array(data2[4::skip])
    l2 = np.array(data2[5::skip])

    xyz1 = np.hstack((x1,y1,z1))
    Rot1 = np.array(R_sb_s1).reshape(3,4)
    Rot1 = Rot1[0:3,0:3]

    xyz2 = np.hstack((x2,y2,z2))
    Rot2 = np.array(R_sb_s2).reshape(3,4)
    Rot2 = Rot2[0:3,0:3]

    u = np.array([])
    v = np.array([])
    w = np.array([])

    length_angle = np.array([])

    for i in range(len(x1)):
        ww = np.array([x1[i],y1[i],z1[i]]).reshape(3,1)
        uvw = np.matmul(Rot1,ww)

        le, th = conv_pc2_length_phi(uvw[0,0],uvw[1,0])
        length_angle = np.append(length_angle, [le, th])

        u = np.append(u,uvw[0,0])
        v = np.append(v,uvw[1,0])
        # w = np.append(w,uvw[2,0])

    for i in range(len(x2)):
        ww = np.array([x2[i],y2[i],z2[i]]).reshape(3,1)
        uvw = np.matmul(Rot2,ww)

        le, th = conv_pc2_length_phi(uvw[0,0],uvw[1,0])
        length_angle = np.append(length_angle, [le, th])

        u = np.append(u,uvw[0,0])
        v = np.append(v,uvw[1,0])
        # w = np.append(w,uvw[2,0])

    return u,v,length_angle

def conv_pc2_length_phi(x,y):
    th = np.arctan2(y,x)
    length = np.sqrt(x*x+y*y)

    return length, th

def Transofrm_point(X,Y,th,x,y):
    xnew = X + x*np.cos(th) - y*np.sin(th)
    ynew = Y + x*np.sin(th) + y*np.cos(th)
    return xnew, ynew

try:
    ani = FuncAnimation(fig, update, frames=800, interval=100, blit=True, repeat=False)
    plt.show()
except:
    sim.stopSimulation()
    f.close()
    
# Stop the simulation
sim.stopSimulation()
f.close()

# main()