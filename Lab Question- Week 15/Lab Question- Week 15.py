import numpy as np
import matplotlib.pyplot as plt

def trapezoidal_rule(f, a, b, n):
  """
  Numerically integrates a function f(x) over the interval [a, b] using the Trapezoidal Rule.

  Args:
    f: The function to integrate.
    a: The lower limit of integration.
    b: The upper limit of integration.
    n: The number of subintervals.

  Returns:
    The approximate value of the definite integral.
  """

  h = (b - a) / n
  x = np.linspace(a, b, n + 1)
  y = f(x)
  integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
  return integral

# Example functions
def polynomial(x):
  return x**2 + 2*x + 1

def trigonometric(x):
  return np.sin(x)

# Define integration limits and number of subintervals
a = 0
b = 2
n = 100

# Calculate the integrals
integral_poly = trapezoidal_rule(polynomial, a, b, n)
integral_trig = trapezoidal_rule(trigonometric, a, b, n)

print("Integral of the polynomial:", integral_poly)
print("Integral of the trigonometric function:", integral_trig)

# Plot the functions and the trapezoidal approximation
x_plot = np.linspace(a, b, 1000)
y_poly_plot = polynomial(x_plot)
y_trig_plot = trigonometric(x_plot)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(x_plot, y_poly_plot, label='Polynomial')
plt.fill_between(x_plot, y_poly_plot, color='lightblue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Integration')

plt.subplot(1, 2, 2)
plt.plot(x_plot, y_trig_plot, label='Trigonometric')
plt.fill_between(x_plot, y_trig_plot, color='lightgreen')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trigonometric Integration')

plt.tight_layout()
plt.show()