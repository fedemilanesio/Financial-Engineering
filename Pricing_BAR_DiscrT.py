import numpy as np
from HESTON_MODEL_INPUTS_BLOOMBERG import r, q, S, StrikeArray_BMB, sigma_NR, GBS, maturities
from HESTON_MODEL_INPUTS_SP500 import prices_SP500, r_sp, divYield_sp, StrikeArray_SP, volatility_sp, spot_sp, maturities_sp
from scipy.stats import norm

##################### DISCRETE TIME/BARRIER #####################
def MC_BARRIER_TERMINAL(N, S, K,Barrier_level,T, r, b, vol, flag):
    H = Barrier_level * S
    discount = np.exp(-r * T)
    underlying_evolution_ITO =S*np.exp((b-(vol**2/2))*T+vol*norm.ppf(np.random.rand(N))*np.sqrt(T))

    flag_hit =(underlying_evolution_ITO < H).astype(int) #terminal observation only

    discounted_payoff = discount * np.maximum(0, underlying_evolution_ITO- K)*flag_hit if flag[0].lower() == "c" else discount * np.maximum(0, K - underlying_evolution_ITO)*flag_hit
    option_price_MC = np.mean(discounted_payoff)
    return option_price_MC

def MC_HE_BARRIER_TERMINAL(N, intervals, params, Barrier_level,optiontype):
    kappa, theta,v0, sigma,rho,lamda = params[0][0], params[0][1], params[0][2],params[0][3],params[0][4], 0
    S, K, T, r, q = params[1][0], params[1][1], params[1][2],params[1][3],params[1][4]
    H = Barrier_level * S
    #kappa = 0
    #sigma = 0
    #v0 = (0.15406416613494095)**2

    sim = np.ones((intervals, N)) * S
    var = np.ones((intervals, N)) * v0
    dt = T / intervals

    Zv = np.random.randn(intervals, N)
    Zs = rho * Zv + np.sqrt(1 - rho ** 2) * np.random.randn(intervals, N)

    for i in range(1, intervals):
        var[i,:] = np.abs(var[i - 1,:] + kappa * (theta - var[i - 1,:]) * dt + sigma * np.sqrt(var[i - 1,:]) * np.sqrt(dt) * Zv[i,:])
        sim[i,:] = sim[i - 1,:] * np.exp((r - q - 0.5 * var[i - 1,:]) * dt + np.sqrt(var[i - 1,:]) * np.sqrt(dt) * Zs[i,:])

    flag_hit =(sim[-1,:] < H).astype(int)

    discount = np.exp(-r * T)
    payoff = np.maximum(0, sim[-1,:] - K) * flag_hit if optiontype[0].lower() == "c" else np.maximum(0,K - sim[-1,:])*flag_hit
    option_price = np.mean(payoff) * discount
    return option_price

### STOCHASTIC TREE
def CRR(S, X, r, b, T, sigma,barrier_level, z, NStep):
    H = barrier_level * S
    dt = T / NStep
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(b * dt) - d) / (u - d)
    OptionValue = np.zeros(NStep + 1)
    x = []
    for i in range(NStep + 1):
        ST = S * u ** i * d ** (NStep - i)
        flag_hit = (ST < H).astype(int)
        OptionValue[i] = np.maximum(0, z * (ST - X))*flag_hit
    for j in range(NStep - 1, -1, -1):
        for i in range(j + 1):
            OptionValue[i] = (p * OptionValue[i + 1] + (1 - p) * OptionValue[i]) * np.exp(-r * dt)
    return OptionValue[0]

########## PARAMETER DEFINITION ##########
T = 3
intervals = 360*T
N = 50000
barrier_level = 0.7
####### BLOOMBERG #######
S_Bl,K_Bl,T_Bl,r_Bl,q_Bl,vol_Bl = S,StrikeArray_BMB[9],T,r[-1],q[-1],sigma_NR[-1,9]
b_Bl = r_Bl - q_Bl
params_BMB = [S_Bl,K_Bl,T_Bl,vol_Bl,r_Bl,q_Bl,b_Bl]

Theta_B = [1.5213994,0.03550201,0.01327494,0.32858323,-0.64150853]
settings_B = [S, StrikeArray_BMB[9], 3, r[-1], q[-1], sigma_NR[-1,9]]

####### SP500 #######

S_SP,K_SP,T_SP,r_SP,q_SP,vol_SP = S,StrikeArray_SP[6], T,r_sp[9],divYield_sp[9], volatility_sp[9,6]
b_SP = r_SP - q_SP
params_SP = [S_SP,K_SP,T_SP,vol_SP,r_SP,q_SP,b_SP]

Theta_S = [0.51196095,0.03452276,0.02226145,0.18800531,-0.90550424]
settings_S = [S, StrikeArray_SP[6], 3, r_sp[9], divYield_sp[9], volatility_sp[9,6]]

##### OUTPUTS ##########
#1) Calling Closed Formula for Pricing BARRIER OPTION
put_MC_Barrier = MC_BARRIER_TERMINAL(50000, S_Bl,K_Bl, barrier_level,T_Bl, r_Bl, b_Bl, vol_Bl,"p")
print("MC ito: ",put_MC_Barrier)
#2) Calling MC HESTON
put_MC_HE_Barrier = MC_HE_BARRIER_TERMINAL(20000,1080,params=[Theta_B,settings_B],Barrier_level=barrier_level,optiontype="put")

print("MC HESTON: ",put_MC_HE_Barrier)

#3) Calling STOCH. TREE
put_CRR_TREE = CRR(S_Bl, K_Bl, r_Bl, b_Bl, T_Bl, vol_Bl, barrier_level,z=-1, NStep=5000)
print("cRR, ",put_CRR_TREE)
pass

put_MC_Barrier = MC_BARRIER_TERMINAL(50000, S_SP,K_SP, barrier_level,T_SP, r_SP, b_SP, vol_SP,"p")
put_MC_HE_Barrier = MC_HE_BARRIER_TERMINAL(20000,1080,params=[Theta_S,settings_S],Barrier_level=barrier_level,optiontype="put")