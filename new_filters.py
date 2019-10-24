import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

thetas = [0,45,90,135]
# theta = math.radians(45.0) # Converts angle x from degrees to radians.

def ChangeBase(x,y,theta):
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = y * math.cos(theta) - x * math.sin(theta)
    return x_theta, y_theta

def GaborFunction(x,y,theta,f,sigma_x,sigma_y):
    r1 = ChangeBase(x,y,theta)[0] / sigma_x
    r2 = ChangeBase(x,y,theta)[1] / sigma_y
    arg = - 0.5 * ( r1**2 + r2**2 )
    return math.exp(arg) * math.cos(2*math.pi*f*ChangeBase(x,y,theta)[0])

def GaborFunction_i(x,y,theta,f,sigma_x,sigma_y):
    r1 = ChangeBase(x,y,theta)[0] / sigma_x
    r2 = ChangeBase(x,y,theta)[1] / sigma_y
    arg = - 0.5 * ( r1**2 + r2**2 )
    return math.exp(arg) * math.sin(2*math.pi*f*ChangeBase(x,y,theta)[0])

def gabor_i(radius, freq, sig_x, sig_y):
    filters_i = []
    for th in thetas:
        M = np.zeros((radius*2+1,radius*2+1))
        x = -float(radius)
        for i in range(radius*2+1):
            y = -float(radius)
            for j in range(radius*2+1):
                M[i,j] = GaborFunction_i(x,y,math.radians(th),freq,sig_x,sig_y)
                y = y + 1
            x = x + 1
        filters_i.append(M)
        # plt.imshow((M + 1) * 128, cmap = cm.Greys_r, origin='lower')
        # plt.savefig('filter/GaborFilter_i'+str(radius*2+1)+'_'+str(th)+'o.png')

    filters_i = np.array(filters_i)
    return filters_i

def gabor(radius, freq, sig_x, sig_y):
    filters = []
    for th in thetas:
        M = np.zeros((radius*2+1,radius*2+1))
        x = -float(radius)
        for i in range(radius*2+1):
            y = -float(radius)
            for j in range(radius*2+1):
                M[i,j] = GaborFunction(x,y,math.radians(th),freq,sig_x,sig_y)
                y = y + 1
            x = x + 1
        filters.append(M)

        # plt.imshow((M + 1) * 128, cmap = cm.Greys_r, origin='lower')
        # plt.savefig('filter/GaborFilter'+str(radius*2+1)+'_'+str(th)+'o.png')
    
    filters = np.array(filters)
    return filters

filter1 = gabor(16, 0.12, 5.1, 5.1)
filter2 = gabor(8, 0.22, 2.7, 2.7)
filter3 = gabor(4, 0.44, 1.45, 1.45)
filter4 = gabor(2, 0.6, 1.25, 1.25)

filter1_i = gabor_i(16, 0.12, 5.1, 5.1)
filter2_i = gabor_i(8, 0.22, 2.7, 2.7)
filter3_i = gabor_i(4, 0.44, 1.45, 1.45)
filter4_i = gabor_i(2, 0.6, 1.25, 1.25)