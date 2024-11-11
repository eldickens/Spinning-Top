import numpy as np
import matplotlib.pyplot as plt

# Constants
I1 = 3 / 4
I3 = 3 / 10
M = 1  # Assume mass M = 1 for simplicity
g = 9.81  # Gravitational constant
l = 1  # Length scale, assume l = 1
damping_coefficient = 0.2  # Damping factor (adjust this value to control the amount of damping)

# Define the system of equations (the RHS of the ODE system)
def fRHS(t, y):
    theta, p_theta, phi, psi, p_phi, p_psi = y

    # Equations of motion
    dtheta_dt = p_theta / I1
    dp_theta_dt = -((p_phi - p_psi * np.cos(theta))**2 / (I1 * np.sin(theta)**3)) + \
                  (p_psi * (p_phi - p_psi * np.cos(theta)) * np.sin(theta) / (I1 * np.sin(theta)**2)) - \
                  M * g * l * np.sin(theta)
    
    # Apply damping to the rate of change of angular momentum
    dp_theta_dt -= damping_coefficient * p_theta  # Damping effect
    
    dphi_dt = (p_phi - p_psi * np.cos(theta)) / (I1 * np.sin(theta)**2)
    dpsi_dt = (p_psi * np.cos(theta) - p_phi) / (I1 * np.sin(theta)**2) + p_psi / I1
    
    dp_phi_dt = 0  # p_phi is constant
    dp_psi_dt = 0  # p_psi is constant

    # Return the derivatives as a vector
    return np.array([dtheta_dt, dp_theta_dt, dphi_dt, dpsi_dt, dp_phi_dt, dp_psi_dt])

# RK4 method 
def rk4(fRHS, t0, y0, dt):
    k1 = fRHS(t0, y0)
    k2 = fRHS(t0 + dt / 2, y0 + k1 * dt / 2)
    k3 = fRHS(t0 + dt / 2, y0 + k2 * dt / 2)
    k4 = fRHS(t0 + dt, y0 + k3 * dt)
    
    # Update the solution
    y = y0 + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    
    return y


# Example scenarios
# 1. Nutation (Original)
theta_0_nutation = np.pi / 4  # Initial nutation angle
p_theta_0_nutation = 1.0  # Initial angular momentum for nutation
phi_0_nutation = 0.0  # Initial precession angle
psi_0_nutation = 0.0  # Initial rotation angle
p_phi_0_nutation = 1.0  # Set p_phi to an arbitrary constant
p_psi_0_nutation = 1.0  # Set p_psi to an arbitrary constant

# Initial state vector for nutation
y0_nutation = np.array([theta_0_nutation, p_theta_0_nutation, phi_0_nutation, psi_0_nutation, p_phi_0_nutation, p_psi_0_nutation])

# 2. Precession
theta_0_precession = 0.05  # Very small initial nutation angle
p_theta_0_precession = 0.0  # No nutation angular momentum
phi_0_precession = 0.0  # Initial precession angle
psi_0_precession = 0.0  # Initial rotation angle
p_phi_0_precession = 5.0  # High precession angular momentum
p_psi_0_precession = 0.1  # Minimal spinning effect

# Initial state vector for precession
y0_precession = np.array([theta_0_precession, p_theta_0_precession, phi_0_precession, psi_0_precession, p_phi_0_precession, p_psi_0_precession])

# 3. Plain Spinning
theta_0_spinning = 1e-3  # Small initial nutation angle to avoid sin(theta) = 0
p_theta_0_spinning = 0.0  # No nutation angular momentum
phi_0_spinning = 0.0  # Initial precession angle
psi_0_spinning = 0.0  # Initial rotation angle
p_phi_0_spinning = 0.1  # Minimal precession effect
p_psi_0_spinning = 10.0  # High spinning angular momentum

# Initial state vector for plain spinning
y0_spinning = np.array([theta_0_spinning, p_theta_0_spinning, phi_0_spinning, psi_0_spinning, p_phi_0_spinning, p_psi_0_spinning])

# You can run the simulation for any of these initial conditions
y0 = y0_nutation


# Time parameters
t0 = 0
tf = 20  # Total time
dt = 0.01  # Time step
n_steps = int((tf - t0) / dt)

# Time array
t = np.linspace(t0, tf, n_steps)

# Initialize solution array
solution = np.zeros((n_steps, len(y0)))
solution[0] = y0

# Run the RK4 integration loop
for i in range(1, n_steps):
    solution[i] = rk4(fRHS, t[i-1], solution[i-1], dt)

# Extract results for plotting
theta_vals = solution[:, 0]
phi_vals = solution[:, 2]
psi_vals = solution[:, 3]

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, theta_vals, label='Theta (Nutation)')
plt.plot(t, phi_vals, label='Phi (Precession)')
plt.plot(t, psi_vals, label='Psi (Rotation)')
plt.xlabel('Time (s)')
plt.ylabel('Angles (radians)')
plt.title('Dynamics of a Symmetrical Spinning Top with Damping')
plt.legend()
plt.grid(True)
plt.show()
