# Programmers: Grant Burk and Ben Croyle
#
# Code packages: numpy, scipy, and matplotlib (see below)
#
# The approach to implement was very simple, just define the expression that we got from doing the math by hand and then get a line space then plot the graph.

import numpy as np
import matplotlib.pyplot as plt

# Define the function
def x(t):
    return np.exp(-t/20), -np.exp(-t/20)

# Create time values
t = np.linspace(0, 100, 400)

# Get the corresponding x(t) values
x1, x2 = x(t)

# Plotting the functions
plt.plot(t, x1, label='$e^{-t/20}$')
plt.plot(t, x2, label='$-e^{-t/20}$')

# Add labels and title
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Plot of $x(t)=[e^{-t/20}, -e^{-t/20}]$')
plt.legend()
plt.grid(True)
plt.show()