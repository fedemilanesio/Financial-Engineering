import numpy as np
from HESTON_MODEL_INPUTS_BLOOMBERG import r, q, S, StrikeArray_BMB, sigma_NR, GBS, maturities
from HESTON_MODEL_INPUTS_SP500 import prices_SP500, r_sp, divYield_sp, StrikeArray_SP, volatility_sp, spot_sp, maturities_sp
from scipy.stats import norm

######## MEMORY EFFECT STRIP COUPON ########
######### CLOSED_FORMULA #########
def MC_DIGITAL_MEMORY_dt(N,freq,params,coupon,optiontype):
    S, K, T, vol, r, q, b = params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    optiontype = 1 if (optiontype.upper()) == 'UPI' else -1
    pay_date = 1 / freq
    t = np.arange(0, T+pay_date, pay_date) ### paym_periods refactored in t since I include 0 as first date.

    underlying_evolution_ITO = np.ones((len(t),N))*S

    condition = np.zeros([len(t)-1,N])
    payoff_matrix = np.zeros([len(t)-1,N])
    df = np.zeros(len(t)-1)

    for i in range(1,len(t)):
        df[i-1] = np.exp(-r * t[i])
        underlying_evolution_ITO[i,:] = underlying_evolution_ITO[i-1,:]*np.exp((b-(vol**2/2))*pay_date+vol*norm.ppf(np.random.rand(N))*np.sqrt(pay_date))
        condition[i-1,:] = (underlying_evolution_ITO[i, :] >= K).astype(int) if optiontype == 1 else (underlying_evolution_ITO[i,:] < K).astype(int)

    for c in range(condition.shape[1]):
        count_missed = 1
        for r in range(condition.shape[0]):
            if condition[r,c] == 0:
                count_missed += 1
                payoff_matrix[r,c] = 0
            else:
                payoff_matrix[r,c] = coupon * condition[r,c] * count_missed * df[r]
                count_missed = 1
    CoN = np.zeros(len(df))
    for i in range(payoff_matrix.shape[0]):
        CoN[i] = np.mean(payoff_matrix[i])
    NPV = np.sum(CoN)
    return f"COUPON MEMORY_EFFECT: {CoN}, NPV: {NPV}"

def MC_HE_DIGITAL_MEMORY(N,intervals,freq,params,coupon,optiontype):
    kappa, theta,v0, sigma,rho,lamda = params[0][0], params[0][1], params[0][2],params[0][3],params[0][4], 0
    S, K, T, r, q = params[1][0], params[1][1], params[1][2],params[1][3],params[1][4]
    optiontype = 1 if (optiontype.upper()) == 'UPI' else -1

    #kappa = 0
    #sigma = 0
    #v0 =  (0.15406416613494095)**2

    pay_date = 1 / freq
    t = np.arange(pay_date, T + pay_date, pay_date)
    dt = T / intervals
    days_count = (t * 360).astype(int)

    # Decomposizione Cholesky
    Zv = np.random.randn(intervals, N)
    Zs = rho * Zv + np.sqrt(1 - rho ** 2) * np.random.randn(intervals, N)

    var = np.ones((intervals, N)) * v0
    sim = np.ones((intervals, N)) * S

    for i in range(1, intervals):
        var[i,:] = np.abs(var[i - 1,:] + kappa * (theta - var[i - 1,:]) * dt + sigma * np.sqrt(var[i - 1,:]) * np.sqrt(dt) * Zv[i,:])
        sim[i,:] = sim[i - 1,:] * np.exp((r - q - 0.5 * var[i - 1,:]) * dt + np.sqrt(var[i - 1,:]) * np.sqrt(dt) * Zs[i,:])

    condition = np.zeros([len(t), N])
    df = np.exp(-r * t)
    sim_for_HE = np.zeros((len(t),N))
    for i, d in enumerate(days_count):
        sim_for_HE[i, :] = sim[d, :]
        condition[i, :] = (sim_for_HE[i, :] >= K).astype(int) if optiontype == 1 else (sim_for_HE[i, :] < K).astype(int)

    payoff_matrix = np.zeros([len(t) , N])
    for c in range(sim_for_HE.shape[1]):
        count_missed = 1
        for r in range(sim_for_HE.shape[0]):
            if condition[r, c] == 0:
                count_missed += 1
                payoff_matrix[r, c] = 0
            else:
                payoff_matrix[r, c] = coupon * condition[r, c] * count_missed * df[r]
                count_missed = 1

    CoN = np.zeros(len(df))
    for i in range(payoff_matrix.shape[0]):
        CoN[i] = np.mean(payoff_matrix[i])
    NPV = np.sum(CoN)
    return f'''COUPON STREAM_MC_HE: {CoN}, NPV: {NPV}'''

########## PARAMETER DEFINITION ##########
########################## BLOOMBERG ##########################
T = 3
intervals = (360 * T) + 1
N = 20000
coupon = 50
NN = 50

S_Bl,K_Bl,T_Bl,r_Bl,q_Bl,vol_Bl = S,StrikeArray_BMB[9],T,r[-1],q[-1],sigma_NR[-1,9]
b_Bl = r_Bl - q_Bl
params_BMB = [S_Bl,K_Bl,T_Bl,vol_Bl,r_Bl,q_Bl,b_Bl]

Theta_B = [1.5213994,0.03550201,0.01327494,0.32858323,-0.64150853]
settings_B = [S, StrikeArray_BMB[9], 3, r[-1], q[-1], sigma_NR[-1,9]]

UPI_MC_Digital_dt_ = MC_DIGITAL_MEMORY_dt(N,2,params_BMB,coupon,"UPI")
UPI_MC_HE_Digital_ = MC_HE_DIGITAL_MEMORY(N, intervals,2, params=[Theta_B, settings_B], coupon=coupon,optiontype="UPI")
print(UPI_MC_Digital_dt_,UPI_MC_HE_Digital_)
pass

########################## SP500 ##########################
S_SP,K_SP,T_SP,r_SP,q_SP,vol_SP = S,StrikeArray_SP[6], T,r_sp[9],divYield_sp[9], volatility_sp[9,6]
b_SP = r_SP - q_SP
params_SP = [S_SP,K_SP,T_SP,vol_SP,r_SP,q_SP,b_SP]

Theta_S = [0.51196095,0.03452276,0.02226145,0.18800531,-0.90550424]
settings_S = [S, StrikeArray_SP[6], 3, r_sp[9], divYield_sp[9], volatility_sp[9,6]]

########## OUTPUT DEFINITION ##########
UP_IN_Digital_dt = MC_DIGITAL_MEMORY_dt(20000,2,params_SP,coupon,"UPI")
UPI_MC_HE_Digital = MC_HE_DIGITAL_MEMORY(20000, intervals,2, params=[Theta_S, settings_S], coupon=coupon,optiontype="UPI")

print(UP_IN_Digital_dt,UPI_MC_HE_Digital)
