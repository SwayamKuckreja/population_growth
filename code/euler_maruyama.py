"""
Created on Tues Jan 28 8:34 pm 2025

@author: kuckreja

@purpose: model stochastic population growth using euler-maruyama scheme

"""

import numpy as np 
import matplotlib.pyplot as plt


def euler_maruyama(c_factor, dt=0.001, T=10.0, num_trials=1000):
    dtc = dt * c_factor  # Coarse time step.
    n = int(T / dt)  # Number of time steps.
    nc = int(T/ dtc)

    sqrtdt = np.sqrt(dt)

    # Model constants
    a1 = 0.5
    a2 = 0.5
    b1 = 1.0
    b2 = 1.0
    be = 1.0
    de = 1.4

    # Error accumulation (for strong error calculation)
    errors = []
    
    # Storing final vals (for weak error calculation)
    y_final: list[float] = []
    yc_final: list[float] = []


    for _ in range(num_trials):  # Run the same dtc alot of times to average
        # Array initialisation
        w1 = np.zeros(n)
        w1c = np.zeros(nc)
        w2 = np.zeros(n)
        w2c = np.zeros(nc)
        w3 = np.zeros(n)
        w3c = np.zeros(nc)
        y = np.zeros(n)
        yc = np.zeros(nc)
        b = np.zeros(n)
        bc = np.zeros(nc)
        d = np.zeros(n)
        dc = np.zeros(nc)

        # IC's
        w1[0]= 0
        w2[0]= 0
        w3[0]= 0
        y[0]= 30
        yc[0]= 30
        b[0] = be
        bc[0] = be
        d[0] = de
        dc[0] = de

        # Run Euler-Maruyama for dt fine
        w1 = np.zeros(n)
        for i in range(n - 1):
            w1[i + 1] = w1[i] + sqrtdt * np.random.randn()
            if (i % c_factor == 0):
                w1c[i//c_factor] = w1[i]



                
                w2c[i//c_factor] = w2[i] 
                w3c[i//c_factor] = w3[i]  # Construct coarse wiener from same fine ones
            y[i+1] = y[i] + (b[i] * y[i] - d[i] * y[i]) * dt + np.sqrt(max(0, b[i] * y[i] + d[i] * y[i])) * (w1[i+1] - w1[i])   
            b[i+1] = b[i] + b1 * (be - b[i]) * dt + a1 * (w2[i+1] - w2[i])   
            d[i+1] = d[i] + b2 * (de - d[i]) * dt + a2 * (w3[i+1] - w3[i])   

        # Run Euler-Maruyama for dt coarse 
        for i in range(nc-1):       
            yc[i+1] = yc[i] + (bc[i] * yc[i] - dc[i] * yc[i]) * dtc + np.sqrt(max(0, bc[i] * yc[i] + dc[i] * yc[i])) * (w1c[i+1] - w1c[i])   
            bc[i+1] = bc[i] + b1 * (be - bc[i]) * dtc + a1 * (w2c[i+1] - w2c[i])   
            dc[i+1] = dc[i] + b2 * (de - dc[i]) * dtc + a2 * (w3c[i+1] - w3c[i])
        
        # Compute strong error using the last value
        error = np.abs(y[-1] - yc[-1])
        errors.append(error)

        # Store the fine and coarse last value
        y_final.append(y[-1])
        yc_final.append(yc[-1])
    

    return dtc, np.mean(errors), np.mean(y_final), np.mean(yc_final)

# Run for different c_factor values
c_factors: list[int] = [2, 5, 10, 20, 50]
strong_errors = []
weak_errors = []
dt_values = []

for c_factor in c_factors:
    dtc, avg_error, avg_y_final, avg_yc_final = euler_maruyama(c_factor)
    dt_values.append(dtc)
    weak_errors.append(np.abs(avg_y_final - avg_yc_final))
    strong_errors.append(avg_error)


# Plot euler-maruyama log-log strong error vs dt
plt.figure(figsize=(8, 6))
plt.loglog(dt_values, strong_errors, 'o-', label='Averaged Strong Error')
plt.loglog(dt_values, np.array(dt_values) ** 0.5, '--', label='Reference Slope (dt^0.5)')
plt.xlabel('Time step size (dt)')
plt.ylabel('Strong Error')
plt.title('Euler-Maruyama Log-Log Strong Error vs dt')
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.show()

# Plot euler-maruyama log-log weak error vs dt
plt.figure(figsize=(8, 6))
plt.loglog(dt_values, weak_errors, 'o-', label='Averaged Weak Error')
plt.loglog(dt_values, np.array(dt_values) ** 1, '--', label='Reference Slope (dt^1)')
plt.xlabel('Time step size (dt)')
plt.ylabel('Weak Error')
plt.title('Euler-Maruyama Log-Log Weak Error vs dt')
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.show()

from scipy.stats import linregress

# Apply log transformation for log-log analysis
log_dt = np.log(dt_values)
log_strong_errors = np.log(strong_errors)
log_weak_errors = np.log(weak_errors)

# Perform linear regression to estimate slope
strong_slope, _, _, _, _ = linregress(log_dt, log_strong_errors)
weak_slope, _, _, _, _ = linregress(log_dt, log_weak_errors)

print(f"Estimated slope for strong error: {strong_slope}")
print(f"Estimated slope for weak error: {weak_slope}")