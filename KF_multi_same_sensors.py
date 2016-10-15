# Kalman filter
from math import *
import numpy as np
from numpy import matrix
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Kalman filter

def kalman_filter(x, P):
    positions = []
    for n in range(len(measurements1)):
            # prediction
            x = (F * x) + u
            P = F * P * F.transpose()
            # measurement update
            Z = matrix([[measurements1[n]],[measurements2[n]]])
            y = Z - (H * x)
            S = H * P * H.transpose() + R
            K = P * H.transpose() * inv(S)
            x = x + (K * y)
            P = (I - (K * H)) * P
            print 'motion matrix'
            print P
            

            positions.append(x[0][0])
    
    print 'R= '
    print(R)
        
    return positions

############################################
############################################


x = matrix([[2.], [0.]]) # initial state (location and velocity)
P = matrix([[1000., 0.], [0., 1000.]]) # initial uncertainty
u = matrix([[0.], [0.]]) # external motion
F = matrix([[1., 1.], [0, 1.]]) # next state function
H = matrix([[1., 0.],[1., 0.]]) # measurement function
I = matrix([[1., 0.], [0., 1.]]) # identity matrix

R = matrix([[2, 0],[0., 10]]) # measurement uncertainty
#sensor data
#sensor data 1
measurements1 = [2, 5, 8]
#sensor data 2
measurements2 = [2, 4, 6]
print('x=',x)
print('P=',P)
if __name__ == '__main__':
    
    positions = kalman_filter(x,P)
    x = np.arange(0,len(measurements1),1)
    positions = np.array(positions)
    print(positions)
    plt.plot(x, positions[:,0],'g', label='Predicted')
    # sensor 1 red
    plt.plot(x, measurements1,'r', label='Sensor 1')
    # sensor 2 blue
    plt.plot(x, measurements2,'b', label='Sensor 2')
    plt.grid()
    
    plt.legend(loc='upper left')
    plt.show()
    print(positions)