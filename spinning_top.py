#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 11:06:45 2024

@author: elenadickens
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
I1 = 3 / 4
I3 = 3 / 10
M = 1  # Assume mass M = 1 for simplicity
g = 9.81  # Gravitational constant
l = 1  # Length scale, assume l = 1

# Define the system of equations (the RHS of the ODE system)
def fRHS(t, y):
    theta, p_theta, phi, psi, p_phi, p_psi = y

    # Equations of motion
    dtheta_dt = p_theta / I1
    dp_theta_dt = -((p_phi - p_psi * np.cos(theta))**2 / (I1 * np.sin(theta)**3)) + \
                  (p_psi * (p_phi - p_psi * np.cos(theta)) * np.sin(theta) / (I1 * np.sin(theta)**2)) - \
                  M * g * l * np.sin(theta)
    
    dphi_dt = (p_phi - p_psi * np.cos(theta)) / (I1 * np.sin(theta)**2)
    dpsi_dt = (p_psi * np.cos(theta) - p_phi) / (I1 * np.sin(theta)**2) + p_psi / I1
    
    dp_phi_dt = 0  # p_phi is constant
    dp_psi_dt = 0  # p_psi is constant

    # Return the derivatives as a vector
    return np.array([dtheta_dt, dp_theta_dt, dphi_dt, dpsi_dt, dp_phi_dt, dp_psi_dt])

# RK4 method as per your professor's structure
def rk4(fRHS, t0, y0, dt):
    k1 = fRHS(t0, y0)
    k2 = fRHS(t0 + dt / 2, y0 + k1 * dt / 2)
    k3 = fRHS(t0 + dt / 2, y0 + k2 * dt / 2)
    k4 = fRHS(t0 + dt, y0 + k3 * dt)
    
    # Update the solution
    y = y0 + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    
    return y

# Initial conditions
theta_0 = np.pi / 4  # Initial nutation angle
p_theta_0 = 0.0  # Initial angular momentum for nutation
phi_0 = 0.0  # Initial precession angle
psi_0 = 0.0  # Initial rotation angle
p_phi_0 = 1.0  # Set p_phi to an arbitrary constant
p_psi_0 = 1.0  # Set p_psi to an arbitrary constant

# Initial state vector
y0 = np.array([theta_0, p_theta_0, phi_0, psi_0, p_phi_0, p_psi_0])

# Time parameters
t0 = 0
tf = 10  # Total time
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
plt.title('Dynamics of a Symmetrical Spinning Top')
plt.legend()
plt.grid(True)
plt.show()
