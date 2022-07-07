# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 17:01:05 2022

@author: Thibault
"""
import heyoka as hy
import numpy as np
from matplotlib.pylab import plt

# Create the symbolic variables.
p, e, i, om, w, f = hy.make_vars("p", "e", "i", "om", "w", "f")

"""
Set parameters.
"""
#Gravitational constant
G_orig = 6.67430 * 10**(-11)
#Solar mass
M_sol = 1.98841 * 10**30

#Using unit conversion to avoid huge numbers 
# mass m' = m/M_0 -> MBH = 1
M_0 = 4.297 * 10**6 * M_sol
# distance r' = r/R_0 -> 1 AU = 1)
R_0 = 149597870700
# time t' = t/T_0 -> 1 time unit ~= 40 minutes (induced by G'=1))
T_0 = np.sqrt((R_0**3)/(G_orig * M_0))

# technically G'
G = 1
#MBH mass (in solar masses)
m1 = 1
# m1 = 8.26*10**36
m2 = 0
#Speed of light (in m/s, then converted) ~= 4.85 AU / 40 minutes
c = 299792458 * T_0 / R_0
#Natural log number
en = 2.718281828459
#Constant that dictates steepness of sigmoid
k = 10000
#Amount of dark matter shells
n = 5



"""
Some common subformulas
"""
M = m1 + m2
GM = G*M
pGM = hy.sqrt(p/GM)
GMCP = GM**2 / (c**2 * p**3)

ecf = e * hy.cos(f)
ecf1 = 1 + ecf
r = p / ecf1

nu = m1 * m2 / (M**2)


"""
Equations (21)-(24):
"""
#(21), (slight alternate form than in paper so we can reuse ecf1)
R1PN = GMCP * ecf1**2 * ((3 * e**2) + 1. + 2 * ecf1  - (4 * ecf**2) \
     + 5 * nu * (1 - (7/19) * e**2) - 8 * nu * hy.cos(f)  + (1/2) * nu * ecf**2)

# Mascon model (mi, ri), sigmoid approximation of step function
# heyoka parameter encoding: [m1,m2,...mn,r1,r2,...rn]
#          ->  par[0..i] for mi, par[n+0..i] for ri
listOfSigs = [0.5 + 0.5 * hy.tanh( k * (r - hy.par[n+i])) for i in range(n)]
listOfRis = [-G * hy.par[i] / (r**2) * listOfSigs[i] for i in range(n)]

#(23)
RDM = hy.sum(listOfRis)

#(24)
R = R1PN + RDM

#(22)
S = GMCP * 2 * (2 - nu) * ecf1**3 * e * hy.sin(f)

W = 0

"""
Osculating equations
"""
#(15)
dpdt = pGM * p * (2/ecf1) * S
#(16)
dedt = pGM * (R * hy.sin(f) + S * (2 * hy.cos(f) + e*(1 + hy.cos(f)**2))/ecf1)
#(17)
didt = pGM * W * hy.cos(w+f)/ecf1
#(18)
domdt = pGM * W * hy.sin(w+f)/(ecf1 * hy.sin(i))
#(19)
#cot = 1/tan
dwdt = pGM * (1/e) * (-R * hy.cos(f) + S * (1. + (1/ecf1) ) * hy.sin(f)  \
                      - W * e * (1/hy.tan(i)) * hy.sin(w+f)/ecf1)
#(20)
dfdt = (1/(pGM*p)) * ecf1**2 + \
        pGM * (1/e) * (R * hy.cos(f) -  S * (1. + (1/ecf1) ) * hy.sin(f))


"""
Instantiate the Taylor integrator
"""
ta = hy.taylor_adaptive(
    # The ODEs.
    [(p, dpdt), (e, dedt), (i, didt), (om, domdt), (w, dwdt), (f, dfdt)],
    # The initial conditions (from https://doi.org/10.1051/0004-6361/202142465 )
    # a = tan(0.12495°/(2*3600)) * 2 * 8277
    # -> p = a * (1-e**2) = 0.00109214 [AU]
    [0.0010921436033369, 0.88441, 134.70, 228.19, 66.25, 0]
)

#TODO: pick a good time limit + step size
t_grid = np.linspace(0, 0.00005, 1000)
#propagate until (for max time) + time grid for plotting


"""
Set dark matter distribution (masses and radii of shells)
"""
# mis = [10**6] + (n-1)*[0]
# mis = n*[10**2]
mis = n*[0]
ta.pars[:n] = mis

ris = np.linspace(0,0.1,n)
ta.pars[n:] = ris


out = ta.propagate_grid(t_grid)

# print(out[4][:, 0])

# Plot some orbital elements
plt.rcParams["figure.figsize"] = (12,6)
plt.subplot(1,2,1)
plt.plot(out[4][:, 0], out[4][:, 1])
plt.xlabel("p")
plt.ylabel("e")
plt.subplot(1,2,2)
plt.plot(out[4][:, 5], out[4][:, 0])
plt.xlabel("p")
plt.ylabel("f");


#Convert to numpy arrays for plotting in 3D with x,y,z
lp  = np.asarray(out[4][:, 0])
le  = np.asarray(out[4][:, 1])
li  = np.asarray(out[4][:, 2])
lom = np.asarray(out[4][:, 3])
lw  = np.asarray(out[4][:, 4])
lf  = np.asarray(out[4][:, 5])

lr = lp / (1 + le * np.cos(lf))



# Position and velocity conversion
rx = lr * (np.cos(lom) * np.cos(lw + lf) - np.cos(li)*np.sin(lom)*np.sin(lw+lf))
ry = lr * (np.sin(lom) * np.cos(lw + lf) + np.cos(li)*np.cos(lom)*np.sin(lw+lf))
rz = lr * np.sin(li) * np.sin(lw + lf)

vx = -np.sqrt(GM)/lp * (np.cos(lom) * (np.sin(lw+lf) + le*np.sin(lw)) + \
         np.cos(li) * np.sin(lom) * (np.cos(lw+lf) + le*np.cos(lw)))
vy = -np.sqrt(GM)/lp * (np.sin(lom) * (np.sin(lw+lf) + le*np.sin(lw)) - \
         np.cos(li) * np.cos(lom) * (np.cos(lw+lf) + le*np.cos(lw)))
vz = np.sqrt(GM)/lp * np.sin(li) * (np.cos(lw+lf) + le * np.cos(lw))



#Plot position and MBH
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(rx, ry, rz, label='Position')
ax.scatter(rx[0], ry[0], rz[0], label='Start',color='lawngreen')
ax.scatter(rx[-1], ry[-1], rz[-1], label='End',color="red")
ax.scatter(0,0,0,color='black',label="MBH")
ax.set_xlabel('rX')
ax.set_ylabel('rY')
ax.set_zlabel('rZ')

#Plot DM shells (3D spheres)
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)

x_sphere = 1 * np.outer(np.cos(u), np.sin(v))
y_sphere = 1 * np.outer(np.sin(u), np.sin(v))
z_sphere = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

for i in range(n):
    #Only plot if mass is not zero
    if mis[i] != 0:
        surf =ax.plot_surface(ris[i]*x_sphere, ris[i]*y_sphere, ris[i]*z_sphere,  \
            rstride=1, cstride=1, color='black', linewidth=0, alpha=0.1,label='DM shell(s)')
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d
        
plotlim = (max(max(abs(rx)),max(abs(ry)),max(abs(rz))))
ax.set_xlim(-plotlim,plotlim)
ax.set_ylim(-plotlim,plotlim)
ax.set_zlim(-plotlim,plotlim)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()


#Plot velocity
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(vx, vy, vz, label='Velocity')
# ax.set_xlabel('vX')
# ax.set_ylabel('vY')
# ax.set_zlabel('vZ')
# ax.legend()
# plt.show()

#Plot dark matter distribution:
# fig = plt.figure()
# plt.scatter(ris, mis)
# plt.xlim(0)
# plt.ylim(0)
# plt.xlabel('Distance from 0')
# plt.ylabel('Mass')
# plt.legend()
# plt.show()











