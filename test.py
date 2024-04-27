import math
import random

import numpy as np
import cv2
def rotateMatrix(location,origin):
    T_location = location/(np.linalg.norm(location))
    T_origin = origin/(np.linalg.norm(origin))
    v = np.multiply(T_location,T_origin)
    c = np.dot(T_location,T_origin)
    I = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,1]])
    V_x = np.array([[0,-v[2],v[1]],
                     [v[2],0,-v[0]],
                     [-v[1],v[0],0]])
    R = I + V_x + np.dot(V_x,V_x)*(1/(1+c))
    return R
if __name__ == '__main__':
    s = np.zeros(shape=(800,800,3))
    print(s.shape)
    #计算外参
    fov = math.pi/3.0
    print(fov)
    width = 800
    height = 800
    result = width/(2*math.tan(fov/2))
    print(result)
    length = [1.0,2.0,3.0]
    print(np.mean(length))
    ##旋转角度转换
    '''forward1 = np.array([0,0,1])
    forward2 = np.array([0,0,1])
    right1 = np.array([1,0,0])
    right2 = right1
    up1 = np.array([0,-1,0])
    up2 = up1
    R1 = rotateMatrix(forward1,forward2)
    R2 = rotateMatrix(right1,right2)
    R3 = rotateMatrix(up1,up2)
    R = np.multiply(np.multiply(R3,R1),R2)
    print(R)
    cv2.SovlePnP()'''



