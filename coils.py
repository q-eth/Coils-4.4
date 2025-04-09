import numpy as np
import matplotlib.pyplot as plt

def magnetic_field_solenoid(x, y, z, x0, y0, z0, length, radius, current, direction):
    mu_0 = 4 * np.pi * 1e-7
    n_turns = 100

    x_rel = x - x0
    y_rel = y - y0
    z_rel = z - z0

    Bx = np.zeros_like(x, dtype=np.float64)
    By = np.zeros_like(y, dtype=np.float64)
    Bz = np.zeros_like(z, dtype=np.float64)
    
    dz = length / n_turns
    
    for i in range(n_turns):
        z_turn = -length / 2 + i * dz
        r = np.sqrt(x_rel**2 + y_rel**2 + (z_rel - z_turn)**2)
        
        dB_magnitude = (mu_0 * current * radius**2) / (2 * (r**2 + radius**2)**(3/2))
        Bx += dB_magnitude * x_rel / r
        By += dB_magnitude * y_rel / r
        Bz += dB_magnitude * (z_rel - z_turn) / r
    
    inside_solenoid = (np.abs(z_rel) <= length / 2) & (np.sqrt(x_rel**2 + y_rel**2) <= radius)
    Bz[inside_solenoid] = mu_0 * current * n_turns / length
    Bx[inside_solenoid] = 0
    By[inside_solenoid] = 0
    
    return direction * Bx, direction * By, direction * Bz

def magnetic_field(x, y, solenoids):
    Bx = np.zeros_like(x, dtype=np.float64)
    By = np.zeros_like(y, dtype=np.float64)
    z = np.zeros_like(x)
    
    for (x0, y0, length, radius, current, direction) in solenoids:
        Bx_solenoid, By_solenoid, _ = magnetic_field_solenoid(x, y, z, x0, y0, 0, length, radius, current, direction)
        Bx += Bx_solenoid
        By += By_solenoid
    
    return Bx, By

def plot_field(solenoids, title):
    x = np.linspace(-0.15, 0.15, 50)
    y = np.linspace(-0.15, 0.15, 50)
    X, Y = np.meshgrid(x, y)
    Bx, By = magnetic_field(X, Y, solenoids)
    
    plt.figure(figsize=(7, 7))
    plt.streamplot(X, Y, Bx, By, color=np.hypot(Bx, By), cmap='plasma', linewidth=1, density=2, arrowstyle='->', arrowsize=1.5)

    plt.title(title)
    plt.xlabel('X (м)')
    plt.ylabel('Y (м)')
    plt.grid(True)
    plt.show()

d = 0.1
length = 0.5
radius = 0.02
current = 1

solenoids_same = [
    (0, d, length, radius, current, 1),
    (0, -d, length, radius, current, 1),
    (d, 0, length, radius, current, 1),
    (-d, 0, length, radius, current, 1)
]

solenoids_quad = [
    (0, d, length, radius, current, -1),
    (0, -d, length, radius, current, -1),
    (d, 0, length, radius, current, 1),
    (-d, 0, length, radius, current, 1)
]

plot_field(solenoids_same, "Magnetic field distribution - all surrents have same direction")
plot_field(solenoids_quad, "Magnetic field distribution - quadrupole")