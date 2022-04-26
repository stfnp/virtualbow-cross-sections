import matplotlib.pyplot as plt 
import numpy as np
from math import sqrt

y_layer = [-0.02, -0.01, 0.01, 0.02]
rho = [0.0]
E = [1e9]

w_max = 0.05
r_back = 0.25
r_belly = 0.1

def width(y):
    # TODO: Check domain of sqrt functions
    w_back = 2*sqrt(r_back**2 - (y - y_layer[-1] + r_back)**2)
    w_belly = 2*sqrt(r_belly**2 - (y - y_layer[0] - r_belly)**2)

    return min(w_max, w_back, w_belly)

    #return 0.05 - 15.0*y**2

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