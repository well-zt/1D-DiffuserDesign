import math
import numpy as np

def NASA7_Cp(T, Highcoeffs, Lowcoeffs,M):
    """
    计算给定温度下的NASA-7多项式拟合的比热容
    
    参数:
    - T: 温度数组 (K)
    - Highcoeffs: 高温系数数组 [a1-a7, Tmin, Tmax]
    - Lowcoeffs: 低温系数数组 [a1-a7, Tmin, Tmax]
    """
    R = 8.314  # 气体常数
    
    # 检查温度范围
    T_min = min(Lowcoeffs[-2], Highcoeffs[-2])
    T_max = max(Lowcoeffs[-1], Highcoeffs[-1])
    
    # 找出越界温度及其位置
    below_min = T < T_min
    above_max = T > T_max
    
    if np.any(below_min) or np.any(above_max):
        error_msg = f"Temperature out of range [{T_min}, {T_max}]K\n"
        if np.any(below_min):
            positions = np.where(below_min)[0]
            temps = T[below_min]
            error_msg += f"Below minimum at positions {positions}, temperatures: {temps}K\n"
        if np.any(above_max):
            positions = np.where(above_max)[0]
            temps = T[above_max]
            error_msg += f"Above maximum at positions {positions}, temperatures: {temps}K"
        raise ValueError(error_msg)
    
    # 创建结果数组
    Cp = np.zeros_like(T)
    
    # 使用低温系数的温度范围
    low_mask = (T >= Lowcoeffs[-2]) & (T <= Lowcoeffs[-1])
    if np.any(low_mask):
        a1, a2, a3, a4, a5, a6, a7 = Lowcoeffs[:7]
        Cp[low_mask] = (a1 + 
                       a2 * T[low_mask] + 
                       a3 * T[low_mask]**2 + 
                       a4 * T[low_mask]**3 + 
                       a5 * T[low_mask]**4) * R/M
    # 使用高温系数的温度范围
    high_mask = (T > Lowcoeffs[-2]) & (T <= Highcoeffs[-1])
    if np.any(high_mask):
        a1, a2, a3, a4, a5, a6, a7 = Highcoeffs[:7]
        Cp[high_mask] = (a1 + 
                        a2 * T[high_mask] + 
                        a3 * T[high_mask]**2 + 
                        a4 * T[high_mask]**3 + 
                        a5 * T[high_mask]**4) * R/M
    print(f"Calculated Cp values: mean = {np.mean(Cp):.2f}, min = {np.min(Cp):.2f}, max = {np.max(Cp):.2f}") 
    return Cp*1000

def calculate_hg(mole_wight_gas, T, kappa, m_dot_g, D_d,Highcoeffs, Lowcoeffs):
    # 计算公式中的各个部分
    eta_g = calculate_eta_g(mole_wight_gas, T)
    Pr_g = 4*kappa/(9*kappa-5)
    cp_g=NASA7_Cp(T, Highcoeffs, Lowcoeffs,mole_wight_gas)
    part1 = 0.026 * (eta_g ** 0.2) / (Pr_g ** 0.6)
    part2 = cp_g * (np.pi / 4) ** (-0.8)
    part3 = (m_dot_g ** 0.8) * (D_d ** (-1.8))

    # 计算最终结果
    h_g = part1 * part2 * part3

    return h_g

def calculate_eta_g(M_bar_g, T_g):
    """
    计算气体的动力粘度 η_{g}
    """
    eta_g = 11.83 * 10**(-8) * (M_bar_g ** 0.5) * (T_g ** 0.6)
    return eta_g

def calculate_hc(U,density_coolant,cp_coolant,lambda_coolant, mu_coolant, D_h_channel):
    """
    计算对流换热系数 h_c

    参数:
    - Re_f: 雷诺数
    - Pr_f: 普朗特数
    - mu_f: 流体的动力粘度 (单位：Pa·s)
    - mu_w: 壁面的动力粘度 (单位：Pa·s)
    - k: 热导率 (单位：W/(m·K))
    - L: 特征长度 (单位：m)

    返回:
    - h_c: 对流换热系数 (单位：W/(m^2·K))
    """
    Pr_f=calculate_Pr(cp_coolant,mu_coolant,lambda_coolant)
    Re_f=calculate_Re(density_coolant,U,D_h_channel,mu_coolant)
    h_c = 0.023 * (Re_f ** 0.8) * (Pr_f ** 0.4) * (lambda_coolant / D_h_channel)
    return h_c

def calculate_Pr(Cp, mu, lambda_):
    """
    计算普朗特数 Pr

    参数:
    - Cp: 比热容 (单位：J/(kg·K))
    - mu: 动力粘度 (单位：Pa·s)
    - lambda_: 热导率 (单位：W/(m·K))

    返回:
    - Pr: 普朗特数
    """
    Pr = (Cp * mu) / lambda_
    return Pr

def calculate_Re(rho, u, D, mu):
    """
    计算雷诺数 Re

    参数:
    - rho: 流体的密度 (单位：kg/m³)
    - u: 流体的流速 (单位：m/s)
    - D: 特征长度 (单位：m)
    - mu: 动力粘度 (单位：Pa·s)

    返回:
    - Re: 雷诺数
    """
    Re = (rho * u * D) / mu
    return Re

def calculate_eta_f(h_c, w_b, k_w, delta_c):
    """
    计算效率 η_f

    参数:
    - w_b: 宽度 (单位：m)
    - k_w: 热导率 (单位：W/(m·K))
    - delta_c: 边界层厚度 (单位：m)

    返回:
    - eta_f: 效率
    """
    # 计算公式中的中间项
    term = math.sqrt(2 * h_c * w_b / k_w) * (delta_c / w_b)

    # 计算 η_f
    eta_f = math.tanh(term) / term

    return eta_f

def calculate_xi(Re):
    """
    计算阻力系数 xi 基于雷诺数 Re

    参数:
    - Re: 雷诺数 (单位：无量纲)

    返回:
    - xi: 阻力系数 (单位：无量纲)
    """
    if 10 <= Re < 4000:
        xi = 0.0025 * Re**0.33
    elif 4000 <= Re < 1e5:
        xi = 0.3164 / Re**0.25
    elif 1e5 <= Re <= 3e6:
        xi = 0.0032 + 0.221 * Re**(-0.237)
    else:
        raise ValueError("Re number out of range")
    return xi

def calculate_delta_p(L, d_e, rho_f, mu_f,U):
    """
    计算压力损失 Δp

    参数:
    - L: 总长度 (单位：m)
    - d_e: 特征直径 (单位：m)
    - rho_f: 流体密度 (单位：kg/m³)
    - nu_f: 流体粘度 (单位：Pa·s)
    - Re: 雷诺数 (单位：无量纲)

    返回:
    - delta_p: 压力损失 (单位：Pa)
    """
    Re = calculate_Re(rho_f, U, d_e, mu_f)
    # print(Re)
    xi = calculate_xi(Re)
    delta_p = xi * (L / d_e) * (rho_f * U**2 / 2)
    return delta_p
def calculate_hc_fin(eta_fin, h_coolant,channel_depth,channel_width,fin_width):
    """
    计算翅片换热系数 h_coolant_fin

    参数:
    - eta_fin: 效率
    - h_coolant: 换热系数

    返回:
    - h_coolant_fin: 翅片换热系数
    """
    correction_factor = (channel_width+2*eta_fin*channel_depth)/(channel_width+fin_width)
    h_coolant_fin = correction_factor * h_coolant
    return h_coolant_fin