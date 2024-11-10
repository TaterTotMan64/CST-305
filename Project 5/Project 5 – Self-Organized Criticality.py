import numpy as np
import matplotlib.pyplot as plt


def lorenz_attractor(x, y, z, dt=0.01, s=10, b=2.667, r=28):
    """
    Generates the Lorenz attractor.

    Args:
        x, y, z: Initial values for x, y, and z.
        dt: Time step.
        s, b, r: Lorenz parameters.

    Returns:
        x, y, z: Arrays containing the time series of x, y, and z values.
    """

    x_list, y_list, z_list = [x], [y], [z]

    for i in range(10000):
        dx = s * (y - x)
        dy = x * (r - z) - y
        dz = x * y - b * z

        x += dx * dt
        y += dy * dt
        z += dz * dt

        x_list.append(x)
        y_list.append(y)
        z_list.append(z)

    return np.array(x_list), np.array(y_list), np.array(z_list)


# Initial values
x0, y0, z0 = 11.8, 4.4, 2.4  # Starting values for the Lorenz attractor

while True:

    # Ask the user if they want to continue or exit
    user_input = input("Enter the r value for the Lorenz attractor (or type 'exit' to quit): ").strip().lower()
    if user_input == 'exit':
        print("Exiting the program. Goodbye!")
        break

    # Prompt the user for the r value
    r_value = float(user_input)

    # Generate the Lorenz attractor
    x, y, z = lorenz_attractor(x0, y0, z0, r=r_value)

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(x, y, z)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title(f"Lorenz Attractor (r={r_value})")

    # Create 2D plots for each pair of axes
    ax2 = fig.add_subplot(222)
    ax2.plot(x, y)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title(f"X vs Y (r={r_value})")

    ax3 = fig.add_subplot(223)
    ax3.plot(x, z)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    ax3.set_title(f"X vs Z (r={r_value})")

    ax4 = fig.add_subplot(224)
    ax4.plot(y, z)
    ax4.set_xlabel("Y")
    ax4.set_ylabel("Z")
    ax4.set_title(f"Y vs Z (r={r_value})")

    plt.tight_layout()
    plt.show()