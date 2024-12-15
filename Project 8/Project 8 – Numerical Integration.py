# Programmers: Grant Burk and Ben Croyle
# Code packages: numpy, matplotlib, and scipy (see below)
# The approach was to define the function and other parameters like number of subintervals and interval range, then use the parameters an function to calculate the Reidmann Sum by adding up all the rectangles of the subintervals

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

print("Part 1, Question a")
print()

# Define the function f(x)
def f(x):
    return np.sin(x) + 1

# Define the interval [-π, π]
x = np.linspace(-np.pi, np.pi, 400)
y = f(x)

# Define the subintervals
subintervals = np.linspace(-np.pi, np.pi, 5)

# Define the width of each subinterval
dx = (np.pi - (-np.pi)) / 4

# Define the left-hand, right-hand, and midpoints
left_endpoints = subintervals[:-1]
right_endpoints = subintervals[1:]
midpoints = (left_endpoints + right_endpoints) / 2

# Calculate the heights of rectangles for each method
heights_left = f(left_endpoints)
heights_right = f(right_endpoints)
heights_mid = f(midpoints)

# Create plots
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

#print the Riemann Sum value
Riemann_sum = np.sum(f(left_endpoints) * dx)
print(f"Left Riemann sum: {Riemann_sum}")

# Plot the function and add rectangles for the left endpoint
axs[0].plot(x, y, 'b', label='f(x) = sin(x) + 1')
axs[0].set_title('Function and Riemann Sum (Left Endpoint)')
axs[0].set_xlabel('x')
axs[0].set_ylabel('f(x)')
for i in range(len(left_endpoints)):
    axs[0].add_patch(plt.Rectangle((left_endpoints[i], 0), dx, heights_left[i], edgecolor='r', facecolor='none'))
axs[0].legend()

#print the Riemann Sum value
Riemann_sum1 = np.sum(f(right_endpoints) * dx)
print(f"Right Riemann sum: {Riemann_sum1}")

# Plot the function and add rectangles for the right endpoint
axs[1].plot(x, y, 'b', label='f(x) = sin(x) + 1')
axs[1].set_title('Function and Riemann Sum (Right Endpoint)')
axs[1].set_xlabel('x')
axs[1].set_ylabel('f(x)')
for i in range(len(right_endpoints)):
    axs[1].add_patch(plt.Rectangle((right_endpoints[i] - dx, 0), dx, heights_right[i], edgecolor='g', facecolor='none'))
axs[1].legend()

#print the Riemann Sum value
Riemann_sum2 = np.sum(f(midpoints) * dx)
print(f"Middle Riemann sum: {Riemann_sum2}")

# Plot the function and add rectangles for the midpoint
axs[2].plot(x, y, 'b', label='f(x) = sin(x) + 1')
axs[2].set_title('Function and Riemann Sum (Midpoint)')
axs[2].set_xlabel('x')
axs[2].set_ylabel('f(x)')
for i in range(len(midpoints)):
    axs[2].add_patch(plt.Rectangle((midpoints[i] - dx/2, 0), dx, heights_mid[i], edgecolor='m', facecolor='none'))
axs[2].legend()

plt.tight_layout()
plt.show()

print()
print()
print("Part 1, Question b")
print()

# Define the function f(x)
def f(x):
    return 3 * x + 2 * x**2

# Define the interval [0, 1]
a = 0
b = 1

# Number of subintervals
n = 10000

# Calculate the width of each subinterval
dx = (b - a) / n

# Calculate the right-hand endpoints
right_endpoints = np.linspace(a + dx, b, n)

# Calculate the Riemann sum using the right-hand endpoints
Riemann_sum = np.sum(f(right_endpoints) * dx)

print(f"Riemann sum: {Riemann_sum}")

# Plot the function and the rectangles
x = np.linspace(a, b, 400)
y = f(x)

fig, ax = plt.subplots()
ax.plot(x, y, 'b', label='f(x) = 3x + 2x^2')

# Add the rectangles for the Riemann sum
for i in range(n):
    ax.add_patch(plt.Rectangle((right_endpoints[i] - dx, 0), dx, f(right_endpoints[i]), edgecolor='r', facecolor='none'))

ax.set_title('Function and Riemann Sum (Right Endpoint)')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()
plt.show()

print()
print()
print("Part 1, Question c:1")

# Define the function f(x)
def f(x):
    return np.log(x)

# Define the interval [1, e]
a = 1
b = np.e

# Number of subintervals (high granularity)
n = 10000

# Calculate the width of each subinterval
dx = (b - a) / n

# Calculate the right-hand endpoints
right_endpoints = np.linspace(a + dx, b, n)

# Calculate the Riemann sum using the right-hand endpoints
Riemann_sum = np.sum(f(right_endpoints) * dx)

print(f"Riemann sum: {Riemann_sum}")

# Plot the function and the rectangles
x = np.linspace(a, b, 400)
y = f(x)

fig, ax = plt.subplots()
ax.plot(x, y, 'b', label='f(x) = ln(x)')

# Add the rectangles for the Riemann sum
for i in range(n):
    ax.add_patch(plt.Rectangle((right_endpoints[i] - dx, 0), dx, f(right_endpoints[i]), edgecolor='r', facecolor='none'))

ax.set_title('Function and Riemann Sum (Right Endpoint)')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()
plt.show()

print()
print()
print("Part 1, Question c:2")
print()

# Define the function f(x)
def f(x):
    return x**2 - x**3

# Define the interval [-1, 0]
a = -1
b = 0

# Number of subintervals (high granularity)
n = 10000

# Calculate the width of each subinterval
dx = (b - a) / n

# Calculate the right-hand endpoints
right_endpoints = np.linspace(a + dx, b, n)

# Calculate the Riemann sum using the right-hand endpoints
Riemann_sum = np.sum(f(right_endpoints) * dx)

print(f"Riemann sum: {Riemann_sum}")

# Plot the function and the rectangles
x = np.linspace(a, b, 400)
y = f(x)

fig, ax = plt.subplots()
ax.plot(x, y, 'b', label='f(x) = x^2 - x^3')

# Add the rectangles for the Riemann sum
for i in range(n):
    ax.add_patch(plt.Rectangle((right_endpoints[i] - dx, 0), dx, f(right_endpoints[i]), edgecolor='r', facecolor='none'))

ax.set_title('Function and Riemann Sum (Right Endpoint)')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()
plt.show()

print()
print()
print("Part 2")
print()

# Example data points
x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
y_data = np.array([141.8, 145, 131, 145.5, 148.9, 144.9, 145.7, 139.1, 137.9, 137.4, 143.7, 149.1, 147.3, 152, 137.8, 139.5, 142.1, 143.6, 140.9, 146.6, 137.9, 147.9, 145.2, 149, 146, 149.4, 152.9, 146.6, 143.3, 148.6])

# convert to MB/min from MB/s
for i in range(len(y_data)):
    y_data[i] = y_data[i] * 60

# Define the power law function
def power_law(x, a1, b1):
    return a1 * x ** b1

# Perform curve fitting
params, covariance = curve_fit(power_law, x_data, y_data)

# Extract parameters
a1, b1 = params

print(f"Fitted parameters: {a1:.4f} * t ^ {b1:.4f}")

# Predict values using the fitted model
y_fit = power_law(x_data, a1, b1)

# Plot the original data and the fitted curve
plt.scatter(x_data, y_data, label='Data points')
plt.plot(x_data, y_fit, color='red', label=f'Fit: y = {a:.2f} * x^{b:.2f}')
plt.xlabel('t(min)')
plt.ylabel('y')
plt.legend()
plt.show()

# Define the function f(x)
def f1(x, a1, b1):
    return a1 * x ** b1

# Define the interval [0, 3]
a = 0
b = 30

# Number of subintervals (high granularity)
n = 10000

# Calculate the width of each subinterval
dx = (b - a) / n

# Calculate the right-hand endpoints
right_endpoints = np.linspace(a + dx, b, n)

# Calculate the Riemann sum using the right-hand endpoints
Riemann_sum = np.sum(f1(right_endpoints, a1, b1) * dx)

print(f"Riemann sum: {Riemann_sum}")

# Plot the function and the rectangles
x = np.linspace(a, b, 400)
y = f1(x, a1, b1)

fig, ax = plt.subplots()
ax.plot(x, y, 'b', label=f'R(t) = {a1} * t^ {b1}')

# Add the rectangles for the Riemann sum
for i in range(n):
    ax.add_patch(plt.Rectangle((right_endpoints[i] - dx, 0), dx, f1(right_endpoints[i], a1, b1), edgecolor='r', facecolor='none'))

ax.set_title('Function and Riemann Sum (Right Endpoint)')
ax.set_xlabel('t')
ax.set_ylabel('R(t)')
ax.legend()
plt.show()
