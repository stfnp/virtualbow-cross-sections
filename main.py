import matplotlib.pyplot as plt 
import numpy as np
from math import sin, cos, sqrt, asin

# Section geometry

class Section:
    def __init__(self, y_layer, w_nominal, r_back_face, r_back_edges, r_belly_face, r_belly_edges):
        self.w_nominal = w_nominal
        
        self.r_back_face = r_back_face
        self.r_back_edges = r_back_edges

        self.r_belly_face = r_belly_face
        self.r_belly_edges = r_belly_edges

        self.y_back = y_layer[-1]
        self.y_belly = y_layer[0]

        if r_back_face > 0:
            self.alpha_back = asin((w_nominal/2 - r_back_edges)/(r_back_face - r_back_edges))
        else:
            self.alpha_back = 0
        
        if r_belly_face > 0:
            self.alpha_belly = asin((w_nominal/2 - r_belly_edges)/(r_belly_face - r_belly_edges))
        else:
            self.alpha_belly = 0
        
        self.y_back_face = self.y_back + r_back_face*(cos(self.alpha_back) - 1)
        self.y_back_edges = self.y_back_face - r_back_edges*cos(self.alpha_back)

        self.y_belly_face = self.y_belly - r_belly_face*(cos(self.alpha_belly) - 1)
        self.y_belly_edges = self.y_belly_face + r_belly_edges*cos(self.alpha_belly)
    
    def width(self, y):
        if self.y_back >= y >= self.y_back_face:
            return self.rounding_back_face(y)
        
        if self.y_back_face >= y >= self.y_back_edges:
            return self.rounding_back_edges(y)
        
        if self.y_back_edges >= y >= self.y_belly_edges:
            return self.w_nominal

        if self.y_belly_face >= y >= self.y_belly:
            return self.rounding_belly_face(y)

        if self.y_belly_edges >= y >= self.y_belly_face:
            return self.rounding_belly_edges(y)
        
        raise ValueError("y out of bounds")

    def rounding_back_face(self, y):
        return self.rounding(0, self.y_back - self.r_back_face, self.r_back_face, y)
    
    def rounding_belly_face(self, y):
        return self.rounding(0, self.r_belly_face - self.y_back, self.r_belly_face, y)

    def rounding_back_edges(self, y):
        if self.r_back_face > 0:
            return self.rounding((self.r_back_face - self.r_back_edges)*sin(self.alpha_back), self.y_back - self.r_back_face + (self.r_back_face - self.r_back_edges)*cos(self.alpha_back), self.r_back_edges, y)
        else:
            return self.rounding(self.w_nominal/2 - self.r_back_edges, self.y_back - self.r_back_edges, self.r_back_edges, y)

    def rounding_belly_edges(self, y):
        if self.r_belly_face > 0:
            return self.rounding((self.r_belly_face - self.r_belly_edges)*sin(self.alpha_belly), self.y_belly + self.r_belly_face - (self.r_belly_face - self.r_belly_edges)*cos(self.alpha_belly), self.r_belly_edges, y)
        else:
            return self.rounding(self.w_nominal/2 - self.r_belly_edges, self.y_belly + self.r_belly_edges, self.r_belly_edges, y)
    
    def rounding(self, x_center, y_center, r, y):
        return 2*(sqrt(r**2 - (y - y_center)**2) + x_center)

y_layer = [-0.02, -0.01, 0.01, 0.02]
w_nominal = 0.05

r_back_face = 0.1
r_back_edges = 0.005

r_belly_face = 0
r_belly_edges = 0.001

section = Section(y_layer, w_nominal, r_back_face, r_back_edges, r_belly_face, r_belly_edges)

# Plot layers
for yi in y_layer:
    w = section.width(yi)
    plt.plot([-w/2, w/2], [yi, yi], "-b")

# Plot width
y_vals = np.linspace(y_layer[0], y_layer[-1], 1000)
w_vals = np.vectorize(section.width)(y_vals)
plt.plot(-w_vals/2, y_vals, "--b")
plt.plot( w_vals/2, y_vals, "--b")

plt.title("Cross Section")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.grid()

plt.show()