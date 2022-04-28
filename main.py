import matplotlib.pyplot as plt 
import numpy as np
from math import sin, cos, sqrt, asin

# Layer boundaries and material properties

y_layer = [-0.02, -0.01, 0.01, 0.02]
rho = [0.0]
E = [1e9]

# Section geometry

w_nominal = 0.05
r_back_center = 0.25
r_back_edges = 0.002

r_belly_center = 0.1
r_belly_edges = 0.005

def positive_arc_x(x_center, y_center, r, y):
    return sqrt(r**2 - (y - y_center)**2) + x_center

def negative_arc_x(x_center, y_center, r, y):
    return -sqrt(r**2 - (y - y_center)**2) + x_center

def width(y):
    y_back = y_layer[-1]
    y_belly = y_layer[0]

    alpha_back_center = asin((w_nominal/2 - r_back_edges)/(r_back_center - r_back_edges))
    alpha_belly_center = asin((w_nominal/2 - r_belly_edges)/(r_belly_center - r_belly_edges))

    y_back_center = y_back + r_back_center*(cos(alpha_back_center) - 1)
    y_back_edges = y_back_center - r_back_edges*cos(alpha_back_center)

    y_belly_center = y_belly - r_belly_center*(cos(alpha_belly_center) - 1)
    y_belly_edges = y_belly_center + r_belly_edges*cos(alpha_belly_center)

    if y_back >= y >= y_back_center:
        return 2*positive_arc_x(0, y_back - r_back_center, r_back_center, y)
    if y_back_center >= y >= y_back_edges:
        return 2*positive_arc_x((r_back_center - r_back_edges)*sin(alpha_back_center), y_back - r_back_center + (r_back_center - r_back_edges)*cos(alpha_back_center), r_back_edges, y)
    if y_belly_edges >= y >= y_belly_center:
        return 2*positive_arc_x((r_belly_center - r_belly_edges)*sin(alpha_belly_center), y_belly + r_belly_center - (r_belly_center - r_belly_edges)*cos(alpha_belly_center), r_belly_edges, y)
    if y_belly_center >= y >= y_belly:
        return 2*positive_arc_x(0, r_belly_center - y_back, r_belly_center, y)
    else:
        return w_nominal

# Plot layers
for yi in y_layer:
    w = width(yi)
    plt.plot([-w/2, w/2], [yi, yi], "-b")

# Plot width
y_vals = np.linspace(y_layer[0], y_layer[-1], 1000)
w_vals = np.vectorize(width)(y_vals)
plt.plot(-w_vals/2, y_vals, "--b")
plt.plot( w_vals/2, y_vals, "--b")

plt.title("Cross Section")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.grid()

plt.show()