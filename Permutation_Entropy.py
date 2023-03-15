
#Sean Lyons
#Thu, Mar 9, 10:11 PM (4 days ago)
#to me
'

import matplotlib.pyplot as plt
import numpy as np
from numpy import std
import itertools
from scipy.integrate import odeint
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
from scipy.special import expit


constant = np.linspace(1, 100, 6000)
def stdfigsize(scale=1, nrows=1, ncols=1, ratio=1.3):
    """
    Returns a tuple to be used as figure size.

    Parameterss
    ----------
    returns (7*ratio*scale*nrows, 7.*scale*ncols)
    By default: ratio=1.3
    ----------
    Returns (7*ratio*scale*nrows, 7.*scale*ncols).
    """

    return((7*ratio*scale*ncols, 7.*scale*nrows))


# Takes data (input positions), dx word length, dy, taux, tauy irrelevent
def ordinal_distribution(data, dx=3, dy=1, taux=1, tauy=1, return_missing=True, tie_precision=None):

    def setdiff(a, b):
        a = np.asarray(a).astype('int64')
        b = np.asarray(b).astype('int64')

        _, ncols = a.shape

        dtype={'names':['f{}'.format(i) for i in range(ncols)],
            'formats':ncols * [a.dtype]}

        C = np.setdiff1d(a.view(dtype), b.view(dtype))
        C = C.view(a.dtype).reshape(-1, ncols)

        return(C)


    try:
        ny, nx = np.shape(data)
        data   = np.array(data)
    except:
        nx     = np.shape(data)[0]
        ny     = 1
        data   = np.array([data])

    if tie_precision is not None:
        data = np.round(data, tie_precision)

    partitions = np.concatenate(
        [
            [np.concatenate(data[j:j+dy*tauy:tauy,i:i+dx*taux:taux]) for i in range(nx-(dx-1)*taux)]
            for j in range(ny-(dy-1)*tauy)
        ]
    )

    symbols = np.apply_along_axis(np.argsort, 1, partitions)
    symbols, symbols_count = np.unique(symbols, return_counts=True, axis=0)

    probabilities = symbols_count/len(partitions)

    if return_missing==False:
        return symbols, probabilities
   
    else:
        all_symbols   = list(map(list,list(itertools.permutations(np.arange(dx*dy)))))
        for i in range(np.math.factorial(dx)):
            if i < len(symbols) and np.array_equal(symbols[i], all_symbols[i]):
                skip=0
            elif i == len(symbols):
                symbols = np.append(symbols, [all_symbols[i]], axis=0)
                probabilities = np.append(probabilities, 0)
            else:
                symbols = np.insert(symbols, i, all_symbols[i], axis=0)
                probabilities = np.insert(probabilities, i, 0)
       
        return symbols, probabilities

def elas_model(L0, k, m, g, tmax, dt, y0):
    def deriv(y, t, L0, k, m):
        """Return the first derivatives of y = theta, z1, L, z2."""
        theta, z1, L, z2 = y

        thetadot = z1
        z1dot = (-g*np.sin(theta) - 2*z1*z2) / L
        Ldot = z2
        z2dot = (m*L*z1**2 - k*(L-L0) + m*g*np.cos(theta)) / m
        return thetadot, z1dot, Ldot, z2dot

   
    t = np.arange(0, tmax+dt, dt)
   

    # Do the numerical integration of the equations of motion
    y = odeint(deriv, y0, t, args=(L0, k, m))
    # Unpack z and theta as a function of time
    theta = y[:,0]

    # Convert to Cartesian coordinates of the two bob positions.
    # x = L * np.sin(theta)
    # y = -L * np.cos(theta)
    return theta

def permutation_entropy(L0, k, m, g, tmax, dt, y0):
    Permutation_Entropy = []
    r = np.linspace(1, 10, 6000)
    for r_ in r:
        datax, datay, datat = elas_model(L0, r_, m, g, tmax, dt, y0)
        n, logistic_probs = ordinal_distribution(datax, return_missing=True)            
        Entropy_Permutation = 0
        for i in range(6):
            logprobs = 0
            logiprob = logistic_probs[i]
            if logiprob == 0:
                logprobs = 0
            else:
                logprobs = np.log(logistic_probs[i])
            Entropy_Permutation += (-1*(logistic_probs[i]*logprobs)/np.log(6))
        Permutation_Entropy.append(Entropy_Permutation)
    return Permutation_Entropy






def permutation_entropy2(L0, k, m, g, tmax, dt, y0, xmin = -5, xmax = 5, ymin = -5, ymax = 5, dots = 100):
    Permutation_Entropy = []
    xcords = []
    ycords = []
    colors = []
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
           starting_y = [starting_theta, y0[1], starting_l, y0[3]]
           datax, datay, datat, datatheta = elas_model(L0, k, m, g, tmax, dt, starting_y)
           n, logistic_probs = ordinal_distribution(datatheta, return_missing=True)            
           Entropy_Permutation = 0
           print('(' + str(len(colors)) + '/' + str(dots**2) + ')')
           for i in range(6):
               logprobs = 0
               logiprob = logistic_probs[i]
               if logiprob == 0:
                   logprobs = 0
               else:
                   logprobs = np.log(logistic_probs[i])
               Entropy_Permutation += (-1*(logistic_probs[i]*logprobs)/np.log(6))
           Permutation_Entropy.append(Entropy_Permutation)
           xcords.append(xcord)
           ycords.append(ycord)
           colors.append(Entropy_Permutation)
    fig1, ax1 = plt.subplots()
    colors = expit(colors)
    plt.scatter(xcords, ycords, c=colors, alpha = 0.1)

    plt.title(str('Permutation entropy as a function of initial conditions. k = ' + str(k)))
       
    #print(colors)
       
    #colors = [color - 10 for color in colors]
       
    #colors = expit(colors)

    print(colors)
       
    plt.scatter(xcords, ycords, c=colors, alpha = 0.2)

       
    print('plot complete')
    return 0
# Pendulum equilibrium spring length (m), spring constant (N.m)
m = 1
# The gravitational acceleration (m.s-2).
g = 9.81
# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 20, .01
# Initial conditions: theta, dtheta/dt, L, dL/dt
y0 = [1.5*np.pi/4, 0, 0.5*L0, 0]


for spring in np.linspace(6.5,0.5,13):
    permutation_entropy2(L0, spring, m, g, tmax, dt, y0)
    plt.show()
    plt.close()








def plotstuff(constant, PE):
    plt.plot(constant,
               PE,
               'o',
               markersize=2,
               color = colors[0],
               label=r'PE Against Spring Constant',
               rasterized=True,
               alpha=0)
    plt.scatter(x=constant, y=PE, s=5)
    plt.xlabel('Spring Constant')
    plt.ylabel('Permutation Entropy',
                     multialignment='center')
    plt.legend(handletextpad=0, markerscale=3)
    #plt.annotate('(0)', (-0.265, 1.2), xycoords='axes fraction', fontsize=20)

L0 = 1
constant = np.linspace(1, 10, 6000)
m = 1
# The gravitational acceleration (m.s-2).
g = 9.81
# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 20, .01
# Initial conditions: theta, dtheta/dt, L, dL/dt
y0 = [2.5*np.pi/4, 3, L0, 1]
correctOrderDX3 = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
logistic_prob_array = [] #012
logistic_prob_array1 = [] #021
logistic_prob_array2 = [] #102
logistic_prob_array3 = [] #120
logistic_prob_array4 = [] #201
logistic_prob_array5 = [] #210
bandt_ra = []
up_down = []
persistence = []
spersistence = []
up_down_scaling = []
rv = []
mh = []
mv = []
rh = []
r = np.linspace(1, 10, 6000)
for r_ in r:
    data = elas_model(L0, r_, m, g, tmax, dt, y0)
    n, logistic_probs = ordinal_distribution(data, return_missing=True)

    logistic_prob_array.append(logistic_probs[0])
    logistic_prob_array1.append(logistic_probs[1])
    logistic_prob_array2.append(logistic_probs[2])
    logistic_prob_array3.append(logistic_probs[3])
    logistic_prob_array4.append(logistic_probs[4])
    logistic_prob_array5.append(logistic_probs[5])
    bandt_ra.append(logistic_probs[2]+logistic_probs[3]-logistic_probs[1]-logistic_probs[4])
    up_down.append(logistic_probs[0]-logistic_probs[5])
    persistence.append(logistic_probs[0]+logistic_probs[5]-(1/3))
    up_down_scaling.append(logistic_probs[1]+logistic_probs[2]-logistic_probs[3]-logistic_probs[4])
    pr1 = (logistic_probs[0]+logistic_probs[3]+logistic_probs[4])/3
    pr2 = (logistic_probs[1]+logistic_probs[2]+logistic_probs[5])/3
    pm1 = (logistic_probs[0]+logistic_probs[5])/2
    pm2 = (logistic_probs[1]+logistic_probs[3])/2
    pm3 = (logistic_probs[2]+logistic_probs[4])/2
    pall = (logistic_probs[0]+logistic_probs[1]+logistic_probs[2]+logistic_probs[3]+logistic_probs[4]+logistic_probs[5])/6
    rv += [((logistic_probs[0]-pr1)**2+(logistic_probs[3]-pr1)**2+(logistic_probs[4]-pr1)**2+(logistic_probs[1]-pr2)**2+(logistic_probs[2]-pr2)**2+(logistic_probs[5]-pr2)**2)/6]
    rh += [((pr1-pall)**2+(pr2-pall)**2)/2]
    mv += [((logistic_probs[0]-pm1)**2+(logistic_probs[5]-pm1)**2+(logistic_probs[1]-pm2)**2+(logistic_probs[3]-pm2)**2+(logistic_probs[2]-pm3)**2+(logistic_probs[4]-pm3)**2)/6]
    mh += [((pm1-pall)**2+(pm2-pall)**2+(pm3-pall)**2)/3]



colors = ['#458B00', '#3888BA', '#000000', '#FF3030', '#FFD700', '#8A3324', '#00EEEE', '#5D478B'
          ]  #palettable.cmocean.diverging.Balance_5_r.hex_colors[1::2]
# Pendulum equilibrium spring length (m), spring constant (N.m)
L0 = 1
m = 1
# The gravitational acceleration (m.s-2).
g = 9.81
# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 20, .01
# Initial conditions: theta, dtheta/dt, L, dL/dt
y0 = [2.5*np.pi/4, 0, L0, 0]

constant = np.linspace(1, 10, 6000)

#PE = permutation_entropy(L0, constant, m, g, tmax, dt, y0)
PE2 = permutation_entropy2(L0, constant, m, tmax, dt, y0)