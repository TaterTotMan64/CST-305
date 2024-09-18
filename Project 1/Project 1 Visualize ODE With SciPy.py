# Programmers: Grant Burk and Ben Croyle
# Code packages: numpy, scipy, and matplotlib (see below)
# The approach to implementing was first to model an ODE in python, in which we choose Newton's law of cooling to depict a CPU cooling. Second, we chose parameters for our system. Lastly, we used odeint to integrate the ODE and plotted it using matplotlib.

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the ODE
def cooling_rate(T, t, k, T_ambient):
    return -k * (T - T_ambient) # Newton's Law of Cooling formula

# Parameters
H = 150 # W/m^2 aluminum heat transfer rate
A = 0.001 # m^2 - estimated cpu surface area
T_ambient = 20  # Ambient temperature
T_initial = 100  # Initial temperature

# Time points
t = np.linspace(0, 30, 100)  # Time in minutes

# Solve the ODE
T = odeint(cooling_rate, T_initial, t, args=(H * A, T_ambient))

# Plot the solution
plt.plot(t, T)
plt.xlabel("Time (minutes)")
plt.ylabel("Temperature (°C)")
plt.title("Temperature of a Cooling Object")
plt.grid(True)
plt.show()