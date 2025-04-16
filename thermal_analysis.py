# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import functions
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


T_coolant_init=300
gas_kappa=1.2
mole_weight_gas=18
inner_diameter=1.3
mass_flow_gas=67.6
wall_thickness=5e-3
solid_conductivity=250
length=1.4 
Highcoeffs=[3.03399, 0.00217692, -1.64073e-07, -9.7042e-11, 1.68201e-14, -30004.3, 4.96677,200,1000]
Lowcoeffs=[4.19864, -0.00203643, 6.5204e-06, -5.48797e-09, 1.77198e-12, -30293.7, -0.849032,1000,3500]


dx=5e-3
mass_flow_coolant=4
channel_width=0.01
channel_depth=0.01
channel_number=80
coolant_density=1000
coolant_lambda=0.60652
coolant_cp=4181.3
coolant_mu=0.00089002

# numerical parameters
max_iterations = 1
min_iterations=100

# Pre-calculation
U=mass_flow_coolant/(coolant_density*channel_depth*channel_width*channel_number)

D_h_channel=2*channel_width*channel_depth/(channel_width+channel_depth)

S=2*np.pi*(inner_diameter+wall_thickness+channel_width)
fin_width=(S-channel_number*channel_width)/channel_number
if fin_width <= 0:
    raise ValueError("Calculated fin width is negative, please check input parameters")

print(f"The hydro diameter is {D_h_channel} m")
print(f"The fin width is {fin_width} m")
print(f"The flow velocity in channel is {U} m/s")


# Read Excel file, skip the first row (index 0)
df = pd.read_excel('ptu.xlsx', skiprows=[0])

# Extract first and second columns and convert to numpy arrays
x = df.iloc[:, 0].to_numpy()
T_stream = df.iloc[:, 1].to_numpy()
T_wall_gas = np.full(len(T_stream), 1300)
T_coolant = np.full(len(T_stream), 300)
T_wall_gas_update = np.full(len(T_stream), 0)

# Iterative calculation
iteration = 0
while True:
    # Calculate film temperature (arithmetic mean)
    T_film = (T_stream + T_wall_gas) / 2
    h_gas=functions.calculate_hg(mole_weight_gas,T_film, gas_kappa, mass_flow_gas, inner_diameter,Highcoeffs, Lowcoeffs)
    q = h_gas * (T_stream-T_wall_gas)
    sum_q = np.cumsum(q)*dx*np.pi*(inner_diameter/2)**2
    T_coolant=sum_q/(coolant_cp*mass_flow_coolant)+T_coolant_init

    h_coolant=functions.calculate_hc(U,coolant_density,coolant_cp, coolant_lambda, coolant_mu, D_h_channel)
    eta_fin=functions.calculate_eta_f(h_coolant,fin_width, solid_conductivity, channel_depth)
    h_coolant_fin=functions.calculate_hc_fin(eta_fin, h_coolant,channel_depth,channel_width,fin_width)
    T_wall_coolant=T_coolant+q/h_coolant_fin
    T_wall_gas_update=q*wall_thickness/solid_conductivity+T_wall_coolant
    
    # Calculate mean squared error
    mse = mean_squared_error(T_wall_gas, T_wall_gas_update)
    
    # Check convergence condition
    if mse <  1e-5 and iteration > min_iterations:
        break
    
    # Update T_wall_gas and increment iteration counter
    T_wall_gas = T_wall_gas_update.copy()
    iteration += 1
    print(f"Iteration {iteration}, MSE: {mse:.2e}")
    # Check if maximum iterations reached
    if iteration >= max_iterations:
        print("Warning: Maximum iterations reached, calculation may not have converged")
        break

Delta_p=functions.calculate_delta_p(length,D_h_channel,coolant_density,coolant_mu,U)*1e6

print(f"the total pressure drop is {Delta_p} Pa")
# 输出迭代次数
print(f"Total iterations: {iteration}")
print(f"the average wall flux is {q.mean()}")

# 在while循环结束后添加绘图代码
plt.figure(figsize=(10, 6))
plt.plot(x, T_wall_gas, 'r-', label='Wall Temperature (Gas Side)')
plt.plot(x, T_coolant, 'b-', label='Coolant Temperature')
# plt.plot(x, T_film, 'g-', label='film Temperature')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.title('Temperature Distribution')
plt.legend()
plt.grid(True)
plt.show()

# 创建双y轴图
fig, ax1 = plt.subplots(figsize=(10, 6))

# 第一个y轴（热传导系数）
ax1.plot(x, h_gas, 'k-', label='Heat Transfer Coefficient')
ax1.set_xlabel('Position (m)')
ax1.set_ylabel('Heat Transfer Coefficient (W/m²K)', color='k')
ax1.tick_params(axis='y', labelcolor='k')

# 第二个y轴（热流密度）
ax2 = ax1.twinx()
ax2.plot(x, q, 'b-', label='Heat Flux')
ax2.set_ylabel('Heat Flux (W/m²)', color='b')
ax2.tick_params(axis='y', labelcolor='b')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('Heat Transfer Coefficient and Heat Flux Distribution')
plt.grid(True)
plt.show()