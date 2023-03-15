#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:51:22 2023

@author: micah
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#from scipy.stats import skew
from scipy.stats import entropy
import collections
from scipy.special import expit

# Pendulum equilibrium spring length (m), spring constant (N.m)
k = 20
m = 1
# The gravitational acceleration (m.s-2).
g = 9.81



L0 = 1
theta0 = -0.01



def deriv(y, t, L0, k, m):
    """Return the first derivatives of y = theta, z1, L, z2."""
    theta, z1, L, z2 = y

    thetadot = z1
    z1dot = (-g*np.sin(theta) - 2*z1*z2) / L
    Ldot = z2
    z2dot = (m*L*z1**2 - k*(L-L0) + m*g*np.cos(theta)) / m
    return thetadot, z1dot, Ldot, z2dot




def find_shannon(starting_angle, L0, k):
    # Maximum time, time point spacings and the time grid (all in s).
    tmax, dt = 20, 0.01
    t = np.arange(0, tmax+dt, dt)
    # Initial conditions: theta, dtheta/dt, L, dL/dt
    y0 = [starting_angle, 0, L0, 0]

    # Do the numerical integration of the equations of motion
    y = odeint(deriv, y0, t, args=(L0, k, m))
    # Unpack z and theta as a function of time
    #theta, L = y[:,0], y[:,2]
    theta = y[:,0]
    # Convert to Cartesian coordinates of the two bob positions.
    #x = L * np.sin(theta)
    #y = -L * np.cos(theta)

    
    theta = np.mod(theta,2*np.pi)
    theta = [round(num,2) for num in theta]
    

    bases = collections.Counter([temporary_base for temporary_base in theta])
        
    distribution = [x/sum(bases.values()) for x in bases.values()]

    return entropy(distribution)


def gradient_plot(xmin, xmax, ymin, ymax, dots, k):
    colors = []
    xcords = []
    ycords = []
    
    for xcord in np.linspace(xmin,xmax,dots):
        
        for ycord in np.linspace(ymin,ymax,dots):
            
            starting_l = np.sqrt(xcord**2 + ycord**2)
            
            if xcord > 0 and ycord < 0:
                starting_theta = np.arctan(-xcord / ycord)
            elif xcord > 0 and ycord > 0:
                starting_theta = np.arctan(ycord / xcord) + np.pi/2
            elif xcord < 0 and ycord > 0:
                starting_theta = np.arctan(-xcord / ycord) + np.pi
            elif xcord < 0 and ycord < 0:
                starting_theta = np.arctan(ycord / xcord) + 1.5*np.pi 
            
            print('(' + str(len(colors)) + '/' + str(dots**2) + ')')
            
            xcords.append(xcord)
            ycords.append(ycord)
            colors.append(find_shannon(starting_theta, starting_l, k))
            
            

    
    
    fig1, ax1 = plt.subplots()
    plt.title(str('Shannon entropy as a function of initial conditions. k = ' + str(k)))
    colors = [color - 10 for color in colors]
    colors = expit(colors)
    plt.scatter(xcords, ycords, c=colors, alpha = 0.2)

    
    print('plot complete')






xarray = []
yarray = []


def entropy_vs_k():
    for spring in np.linspace(5,100,101):
    xarray.append(spring)
    yarray.append(find_shannon(2.5 * np.pi/4, 1,spring))
    print(spring)

    plt.scatter(xarray,yarray)
    plt.xlabel('k')
    plt.ylabel('Shannon Entropy')



#Our main function for creating the gradient plot

for spring in np.linspace(21,21,1):
    gradient_plot(-1, 1, -1, 1, 100, spring)
    plt.show()
    plt.close()



