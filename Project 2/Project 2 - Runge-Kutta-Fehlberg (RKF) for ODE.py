# Programmers: Grant Burk and Ben Croyle
#
# Code packages: numpy, scipy, and matplotlib (see below)
#
# The approach to implementing was to use the Runge Kutta method to solve ODE.
# The formula was taken from the book and used to solve the ODE: y / (np.exp(x) - 1). The Runge Kutta method's results were then compared
# using a library to solve the ODE.


import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate


def f(x, y):
    """
    Defines the right-hand side of the differential equation.

    Args:
        x: Independent variable.
        y: Dependent variable.

    Returns:
        The value of dy/dx.
    """
    return y / (np.exp(x) - 1)


def runge_kutta_4(f, x0, y0, h, n, print_steps=True):
    """
    Solves the differential equation using the 4th-order Runge-Kutta method.

    Args:
        f: The function defining the right-hand side of the ODE.
        x0: The initial x value.
        y0: The initial y value.
        h: The step size.
        n: The number of steps.

    Returns:
        A list of x values and a list of corresponding y values.
    """
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0] = x0
    y[0] = y0

    for i in range(n):
        if (print_steps):
            print("For n=" + str(i) + ": ")
            print()

        # Solve for K1
        k1 = h * f(x[i], y[i])
        # Show steps
        if (print_steps):
            print("k1 = h * f(x[n], y[n])")
            print("k1 = " + str(h) + " * f(" + str(x[i]) + ", " + str(y[i]) + ")")
            print("k1 = " + str(h) + " * " + str(f(x[i], y[i])))
            print("k1 = " + str(h * f(x[i], y[i])))
            print()

        # Solve for K2
        k2 = h * f(x[i] + h / 2, y[i] + k1 / 2)
        # Show steps
        if (print_steps):
            print("k2 = h * f(x[n] + h/2, y[n] + k1/2)")
            print(
                "k2 = " + str(h) + " * f(" + str(x[i]) + " + " + str(h) + "/2, " + str(y[i]) + " + " + str(k1) + "/2)")
            print("k2 = " + str(h) + " * f(" + str(x[i]) + " + " + str(h / 2) + ", " + str(y[i]) + " + " + str(
                k1 / 2) + ")")
            print("k2 = " + str(h) + " * f(" + str(x[i] + h / 2) + ", " + str(y[i] + k1 / 2) + ")")
            print("k2 = " + str(h) + " * " + str(f(x[i] + h / 2, y[i] + k1 / 2)))
            print("k2 = " + str(h * f(x[i] + h / 2, y[i] + k1 / 2)))
            print()

        # Solve for K3
        k3 = h * f(x[i] + h / 2, y[i] + k2 / 2)
        # Show steps
        if (print_steps):
            print("k3 = h * f(x[n] + h/2, y[n] + k2/2)")
            print(
                "k3 = " + str(h) + " * f(" + str(x[i]) + " + " + str(h) + "/2, " + str(y[i]) + " + " + str(k2) + "/2)")
            print("k3 = " + str(h) + " * f(" + str(x[i]) + " + " + str(h / 2) + ", " + str(y[i]) + " + " + str(
                k2 / 2) + ")")
            print("k3 = " + str(h) + " * f(" + str(x[i] + h / 2) + ", " + str(y[i] + k2 / 2) + ")")
            print("k3 = " + str(h) + " * " + str(f(x[i] + h / 2, y[i] + k2 / 2)))
            print("k3 = " + str(h * f(x[i] + h / 2, y[i] + k2 / 2)))
            print()

        # Solve for K4
        k4 = h * f(x[i] + h, y[i] + k3)
        # Show steps
        if (print_steps):
            print("k4 = h * f(x[n] + h, y[n] + k3)")
            print("k4 = " + str(h) + " * f(" + str(x[i]) + " + " + str(h) + ", " + str(y[i]) + " + " + str(k3) + ")")
            print("k4 = " + str(h) + " * f(" + str(x[i] + h) + ", " + str(y[i] + k3) + ")")
            print("k4 = " + str(h) + " * " + str(f(x[i] + h, y[i] + k3)))
            print("k4 = " + str(h * f(x[i] + h, y[i] + k3)))
            print()

        # Solve for x[i+1]
        x[i + 1] = x[i] + h
        # Show steps
        if (print_steps):
            print("x[i+1] = x[i] + h")
            print("x[i+1] = " + str(x[i]) + " + " + str(h))
            print("x[i+1] = " + str(x[i] + h))
            print()

        # Solve for y[i+1]
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # Show steps
        if (print_steps):
            print("y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6")
            print("y[i+1] = " + str(y[i]) + " + (" + str(k1) + " + 2*" + str(k2) + " + 2*" + str(k3) + " + " + str(
                k4) + ") / 6")
            print("y[i+1] = " + str(y[i]) + " + (" + str(k1) + " + " + str(2 * k2) + " + " + str(2 * k3) + " + " + str(
                k4) + ") / 6")
            print("y[i+1] = " + str(y[i]) + " + (" + str(k1 + 2 * k2 + 2 * k3 + k4) + ") / 6")
            print("y[i+1] = " + str(y[i]) + " + " + str((k1 + 2 * k2 + 2 * k3 + k4) / 6))
            print("y[i+1] = " + str(y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6))
            print()

    return x, y


# initialize ODE variable
x0 = 1
y0 = 5
h = 0.02
n = 1499  # make 1500 points of data

# call the Runge Kutta solving function with the ODE and variables
x_values, y_values = runge_kutta_4(f, x0, y0, h, n, False)  # Solve using Runge Kutta Method

# format y0 to work in the new function
y0 = [5]

# Create the time points for integration
t_span = (1, 31)

# Define the points at which to store the computed solution
t_eval = np.linspace(1, 31, 1500)  # make 1500 points of data

# Solve the differential equation using solve_ivp
sol = integrate.solve_ivp(f, t_span, y0, t_eval=t_eval)

# Extract the solution
x_sol = sol.t
y_sol = sol.y[0]

# Print the values from the Runge Kutta Method
print("Runge Kutta Method")
print("x\t\ty")
for x, y in zip(x_values, y_values):
    print(f"{x:.2f}\t{y:.4f}")

# Print the values from using the scipy.integrate.solve_ivp Method
print("scipy.integrate.solve_ivp Method")
print("x\t\ty")
for x, y in zip(x_sol, y_sol):
    print(f"{x:.2f}\t{y:.4f}")

# Compare both methods side by side to see any differences
print("Compare:")
print("x\t\tx_actual\ty\t\ty_actual")
for i in range(len(x_values)):
    print(f"{x_values[i]:.2f}\t{x_sol[i]:.2f}\t\t{y_values[i]:.2f}\t{y_sol[i]:.4f}")

# Plot the results of the Runge Kutta Method
plt.plot(x_values, y_values, label='Runge-Kutta 4')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Plot the results of the scipy.integrate.solve_ivp Method
plt.plot(x_sol, y_sol, label='solve_ivp')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Plot the results from both methods on the same plot
plt.plot(x_values, y_values, label='Runge-Kutta 4')
plt.plot(x_sol, y_sol, label='solve_ivp')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Calculate the difference in the y values as the x values are the same
y_dif = []
for i in range(len(x_values)):
    y_dif.append(x_values[i] - x_sol[i])
    # Print the difference in values
    print("Difference between y values: ", y_dif[i])

# Plot different in y value in respects to x
plt.plot(x_values, y_dif, label='Y Difference')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()