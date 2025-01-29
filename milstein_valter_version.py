"""
Created on Tues Jan 28 9:33 pm 2025

@author: kuckreja

@purpose: model stochastic population growth using valters version of milstein scheme 

"""

import numpy as np 
import matplotlib.pyplot as plt


def milstein(c_factor, dt=0.001, T=1.0, num_trials=1000):  # Set simulation time, fine dt and number of trials before averaging
    dtc = dt * c_factor  # Coarse time step
    n = int(T / dt)  # Fine time steps
    nc = int(T / dtc)  # Coarse time steps

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
    

    for q in range(num_trials):  # Run the same dtc alot of times to average

        if q%1000==0:
            print(q)

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

        # Run Milstein for fine dt
        for i in range(n - 1):
            w1[i + 1] = w1[i] + sqrtdt * np.random.randn()
            w2[i + 1] = w2[i] + sqrtdt * np.random.randn()
            w3[i + 1] = w3[i] + sqrtdt * np.random.randn()
            if (i % c_factor == 0):
                w1c[i//c_factor] = w1[i]
                w2c[i//c_factor] = w2[i] 
                w3c[i//c_factor] = w3[i]  # Construct coarse wiener from same fine ones

            sqrt_term = np.sqrt(max(0, b[i] * y[i] + d[i] * y[i]))

            # Here valters derivation for the milstein scheme is used
            y[i + 1] = y[i] + (b[i] * y[i] - d[i] * y[i]) * dt + sqrt_term * (w1[i+1] - w1[i]) + 0.25*(((b[i]+d[i])*(((w1[i+1] - w1[i])**2) - dt)) + ((a1*y[i]*(w2[i+1] - w2[i])*(w1[i+1] - w1[i]))/sqrt_term) + ((a2*y[i]*(w3[i+1] - w3[i])*(w1[i+1] - w1[i]))/sqrt_term))
            b[i + 1] = b[i] + b1 * (be - b[i]) * dt + a1 * (w2[i+1] - w2[i]) 
            d[i + 1] = d[i] + b2 * (de - d[i]) * dt + a2 * (w3[i+1] - w3[i])  
        

        # Run Milstein for dt coarse
        for i in range(nc - 1):
            sqrt_term_c = np.sqrt(max(0, bc[i] * yc[i] + dc[i] * yc[i]))

            yc[i + 1] = yc[i] + (bc[i] * yc[i] - dc[i] * yc[i]) * dtc + sqrt_term_c * (w1c[i + 1] - w1c[i]) + 0.25*(((bc[i]+dc[i])*(((w1c[i+1] - w1c[i])**2) - dtc)) + ((a1*yc[i]*(w2c[i+1] - w2c[i])*(w1c[i+1] - w1c[i]))/sqrt_term_c) + ((a2*yc[i]*(w3c[i+1] - w3c[i])*(w1c[i+1] - w1c[i]))/sqrt_term_c))
            bc[i + 1] = bc[i] + b1 * (be - bc[i]) * dtc + a1 * (w2c[i + 1] - w2c[i])
            dc[i + 1] = dc[i] + b2 * (de - dc[i]) * dtc + a2 * (w3c[i + 1] - w3c[i])

        # Compute strong error using last time step
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
    dtc, avg_error, avg_y_final, avg_yc_final = milstein(c_factor)
    dt_values.append(dtc)
    weak_errors.append(np.abs(avg_y_final - avg_yc_final))
    strong_errors.append(avg_error)


# Plot milstein log-log strong error vs dt
plt.figure(figsize=(8, 6))
plt.loglog(dt_values, strong_errors, 'o-', label='Averaged Strong Error')
plt.loglog(dt_values, np.array(dt_values) ** 0.5, '--', label='Reference Slope (dt^0.5)')
plt.xlabel('Time step size (dt)')
plt.ylabel('Strong Error')
plt.title('Milstein Log-Log Strong Error vs dt')
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.show()

# Plot milstein log-log weak error vs dt
plt.figure(figsize=(8, 6))
plt.loglog(dt_values, weak_errors, 'o-', label='Averaged Weak Error')
plt.loglog(dt_values, np.array(dt_values) ** 1, '--', label='Reference Slope (dt^1)')
plt.xlabel('Time step size (dt)')
plt.ylabel('Weak Error')
plt.title('Milstein Log-Log Weak Error vs dt')
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.show()
