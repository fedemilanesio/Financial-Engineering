import numpy as np
from HESTON_MODEL_INPUTS_BLOOMBERG import r, q, S, StrikeArray_BMB, sigma_NR, GBS, maturities
from HESTON_MODEL_INPUTS_SP500 import prices_SP500, r_sp, divYield_sp, StrikeArray_SP, volatility_sp, spot_sp, maturities_sp
from scipy.stats import norm

######### CONTINUOUS TIME/BARRIER#########
######### BLOOMBERG DATA#########
######### CLOSED_FORMULA #########

def BARRIER_CLOSED(params,option_type, barrier_type,barrier_level):
    S, K, T, sigma, r, q = params[0],params[1],params[2],params[3],params[4],params[5]
    b = r-q
    H = barrier_level * S
    rebate = 0
    option_type = option_type[0].upper()
    match (option_type, barrier_type):
        case ("C", "DI"):
            eta, phi = 1, 1
        case ("C", "UI"):
            eta, phi = -1, 1
        case ("P", "UI"):
            eta, phi = -1, -1
        case ("P", "DI"):
            eta, phi = 1, -1
        case ("C", "DO"):
            eta, phi = 1, 1
        case ("C", "UO"):
            eta, phi = -1, 1
        case ("P", "UO"):
            eta, phi = -1, -1
        case ("P", "DO"):
            eta, phi = 1, -1
        case _:
            raise ValueError("Invalid Option or Barrier Type")
    mu = (b - sigma ** 2 / 2) / sigma ** 2
    x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    y1 = np.log(H ** 2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    lam = np.sqrt(mu ** 2 + 2 * r / sigma ** 2)
    z = np.log(H / S) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)

    A = phi * S * np.exp((b - r) * T) * norm.cdf(phi * x1) - phi * K * np.exp(-r * T) * norm.cdf(
        phi * x1 - phi * sigma * np.sqrt(T))
    B = phi * S * np.exp((b - r) * T) * norm.cdf(phi * x2) - phi * K * np.exp(-r * T) * norm.cdf(
        phi * x2 - phi * sigma * np.sqrt(T))
    C = phi * S * np.exp((b - r) * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(eta * y1) - phi * K * np.exp(-r * T) * (
                H / S) ** (2 * mu) * norm.cdf(eta * y1 - eta * sigma * np.sqrt(T))
    D = phi * S * np.exp((b - r) * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(eta * y2) - phi * K * np.exp(-r * T) * (
                H / S) ** (2 * mu) * norm.cdf(eta * y2 - eta * sigma * np.sqrt(T))
    E = rebate * np.exp(-r * T) * (norm.cdf(eta * x2 - eta * sigma * np.sqrt(T)) - ((H / S) ** (2 * mu)) * norm.cdf(
        eta * y2 - eta * sigma * np.sqrt(T)))
    F = rebate * ((H / S) ** (mu + lam) * norm.cdf(eta * z) + (H / S) ** (mu - lam) * norm.cdf(
        eta * z - 2 * eta * lam * sigma * np.sqrt(T)))
    value = 0
    if (option_type == "C") and (barrier_type == "DI"):
        value = C + E if (K > H) else A - B + D + E
    if (option_type == "C") and (barrier_type == "UI"):
        value = A + E if (K > H) else B - C + D + E
    if (option_type == "P") and (barrier_type == "DI"):
        value = B - C + D + E if (K > H) else A + E
    if (option_type == "P") and (barrier_type == "UI"):
        value = A - B + D + E if (K > H) else C + E
    if (option_type == "C") and (barrier_type == "DO"):
        value = A - C + F if (K > H) else B - D + F
    if (option_type == "C") and (barrier_type == "UO"):
        value = F if (K > H) else A - B + C - D + F
    if (option_type == "P") and (barrier_type == "DO"):
        value = A-B+C-D+F if (K > H) else F
    if (option_type == "P") and (barrier_type == "UO"):
        value = B-D+F if (K > H) else A-C+F
    return value

#########  MONTECARLO_SIMULATIONS #########
# 1) "A LA BLACK-SCHOLES"
def MC_BARRIER(N, intervals, S, K,Barrier_level,T, r, b, vol, flag):
    H = Barrier_level * S
    dt = T/intervals
    discount = np.exp(-r * T)
    underlying_evolution_ITO = np.ones((intervals,N)) * S
    flag_hit=np.zeros(N)
    for i in range(1,intervals):
        underlying_evolution_ITO[i,:] = underlying_evolution_ITO[i-1,:]*np.exp((b-(vol**2/2))*dt+vol*norm.ppf(np.random.rand(N))*np.sqrt(dt))
    for j in range(1,N):
        flag_hit[j] = (min(underlying_evolution_ITO[:,j] ) < H).astype(int)

    discounted_payoff = discount * np.maximum(0, underlying_evolution_ITO[-1, :] - K)*flag_hit if (flag.upper()).startswith("C") else discount * np.maximum(0, K - underlying_evolution_ITO[-1, :])*flag_hit
    option_price_MC = np.mean(discounted_payoff)
    return option_price_MC

# 2) HESTON IN MC
def MC_HE_BARRIER(N, intervals, params, Barrier_level, optiontype):
    kappa, theta,v0, sigma,rho,lamda = params[0][0], params[0][1], params[0][2],params[0][3],params[0][4], 0
    S, K, T, r, q = params[1][0], params[1][1], params[1][2],params[1][3],params[1][4]
    H = Barrier_level * S

    #kappa = 0
    #sigma = 0
    #v0 = ( sigma_NR[-1,9])**2 ---> for bloomberg
    #v0 = volatility_sp[9, 6]**2 ----> for SP500

    sim = np.ones((intervals, N)) * S
    var = np.ones((intervals, N)) * v0
    dt = T / intervals

    Zv = np.random.randn(intervals, N)
    Zs = rho * Zv + np.sqrt(1 - rho ** 2) * np.random.randn(intervals, N)
    flag_hit = np.zeros(N)
    for i in range(1, intervals):
        var[i,:] = np.abs(var[i - 1,:] + kappa * (theta - var[i - 1,:]) * dt + sigma * np.sqrt(var[i - 1,:]) * np.sqrt(dt) * Zv[i,:])
        sim[i,:] = sim[i - 1,:] * np.exp((r - q - 0.5 * var[i - 1,:]) * dt + np.sqrt(var[i - 1,:]) * np.sqrt(dt) * Zs[i,:])

    for j in range(1,N):
        flag_hit[j] = (min(sim[:,j]) < H).astype(int)
    discount = np.exp(-r * T)
    payoff = np.maximum(0, sim[-1,:] - K) * flag_hit if (optiontype.upper()).startswith("C") else np.maximum(0,K - sim[-1,:])*flag_hit
    option_price = np.mean(payoff) * discount
    return option_price

########## PARAMETER DEFINITION ##########
T = 3
intervals = 360*T
N = 50000
barrier_level = 0.7

###### BLOOMBERG   #######
Theta_B = [1.5213994,0.03550201,0.01327494,0.32858323,-0.64150853]
settings_B = [S, StrikeArray_BMB[9], 3, r[-1], q[-1], sigma_NR[-1,9]]

S_Bl,K_Bl,T_Bl,r_Bl,q_Bl,vol_Bl = S,StrikeArray_BMB[9],T,r[-1],q[-1],sigma_NR[-1,9]
b_Bl = r_Bl - q_Bl
params_BMB = [S_Bl,K_Bl,T_Bl,vol_Bl,r_Bl,q_Bl]

###### SP500  #######
Theta_S = [0.51196095,0.03452276,0.02226145,0.18800531,-0.90550424]
settings_S = [S, StrikeArray_SP[6], 3, r_sp[9], divYield_sp[9], volatility_sp[9,6]]

S_SP,K_SP,T_SP,r_SP,q_SP,vol_SP = S,StrikeArray_SP[6], T,r_sp[9],divYield_sp[9], volatility_sp[9,6]
b_SP = r_SP - q_SP
params_SP = [S_SP,K_SP,T_SP,vol_SP,r_SP,q_SP]
############################################## CALLING OUTPUTS #####################################################
#1) Calling Closed Formula for Pricing BARRIER OPTION
#call_Barrier = BARRIER_CLOSED(params_BMB,"C","DI",barrier_level)
put_Barrier_b= BARRIER_CLOSED(params_BMB,"Put","DI",barrier_level)
put_Barrier_s = BARRIER_CLOSED(params_SP,"Put","DI",barrier_level)
print("BARRIER_OPTION CLOSED FORMULA b: ", put_Barrier_b)
print("BARRIER_OPTION CLOSED FORMULA s: ", put_Barrier_s)
#2) Calling MC Traditional
#call_MC_Barrier = MC_BARRIER(50000, 1080, S_Bl,K_Bl, barrier_level,T_Bl, r_Bl, b_Bl, vol_Bl,"c")
#put_MC_Barrier = MC_BARRIER(20000, 1080, S_Bl,K_Bl, barrier_level,T_Bl, r_Bl, b_Bl, vol_Bl,"p")
#put_MC_Barrier = MC_BARRIER(20000, 1080, S_SP,K_SP, barrier_level,T_SP, r_SP, b_SP, vol_SP,"p")
#test_B_bar = []
#[test_B_bar.append(MC_BARRIER(20000, 1080, S_SP,K_SP, barrier_level,T_SP, r_SP, b_SP, vol_SP,"p")) for i in range(50)]
#print("MONTECARLO CONVERGENCE CHECK: ", put_MC_Barrier)

#2) Calling MC with HESTON
put_MC_HE_Barrier_b = MC_HE_BARRIER(20000, 1080, params=[Theta_B, settings_B], Barrier_level=barrier_level, optiontype="put")
put_MC_HE_Barrier_s = MC_HE_BARRIER(20000, 1080, params=[Theta_S, settings_S], Barrier_level=barrier_level, optiontype="put")
print("MONTECARLO HESTON b: ", put_MC_HE_Barrier_b)
print("MONTECARLO HESTON s: ", put_MC_HE_Barrier_s)

# CHECK 1) GBS vs BARRIER OPTION WITH A BARRIER ~ IMPOSSIBLE TO BE HIT
put_Barrier_check = BARRIER_CLOSED(params_BMB,"P","DI",2e10) # 440.994
put_BS_check = GBS(S_Bl,K_Bl,T_Bl,r_Bl,r_Bl - q_Bl,vol_Bl,'P') # 440.994
## Same Check with Heston by setting kappa,theta = 0
############### MONTECARLO WITH dt INSTEAD OF T, THAT LEADS TO THE SAME RESULSTS
'''
def MC_dt(N, intervals, S, K,T, r, b, vol, flag):
    dt = T/intervals
    discount = np.exp(-r * T)
    underlying_evolution_ITO = np.ones((intervals,N)) * S
    for i in range(1,intervals):
        underlying_evolution_ITO[i,:] = underlying_evolution_ITO[i-1,:]*np.exp((b-(vol**2/2))*dt+vol*norm.ppf(np.random.rand(N))*np.sqrt(dt))

    discounted_payoff = discount * np.maximum(0, underlying_evolution_ITO[-1, :] - K) if flag.lower() == "c" else discount * np.maximum(0, K -underlying_evolution_ITO[-1, :])
    option_price_MC = np.mean(discounted_payoff)
    return option_price_MC
'''