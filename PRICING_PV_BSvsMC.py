import numpy as np
from HESTON_MODEL_INPUTS_BLOOMBERG import r, q,S, StrikeArray_BMB, sigma_NR,GBS
from HESTON_MODEL_INPUTS_SP500 import prices_SP500,r_sp, divYield_sp, StrikeArray_SP, volatility_sp,spot_sp,maturities_sp
from scipy.stats import norm
import matplotlib.pyplot as plt
def MC_BS(N, S, K, T, r, b, vol, flag):
    discount = np.exp(-r * T)
    underlying_evolution_ITO = S*np.exp((b-(vol**2/2))*T+vol*norm.ppf(np.random.rand(N))*np.sqrt(T))
    discounted_payoff = discount * np.maximum(0, underlying_evolution_ITO - K) if flag.lower() == "c" else discount * np.maximum(0, K-underlying_evolution_ITO)
    option_price_MC = np.mean(discounted_payoff)
    return option_price_MC

N_repl = 10000
N = 20000

############## CALL ################
## BS
### BLOOMBERG
K,T,r,b,vol,flag = StrikeArray_BMB[9],3,r[-1], r[-1] - q[-1],sigma_NR[-1,9], "c"
call_BS_formula_BMB= GBS(S,K,T,r,b,vol,flag)
print("BS", call_BS_formula_BMB)
test_B = []
call_BS_MC = MC_BS(N, S, K, T, r, b, vol, flag)
[test_B.append(MC_BS(N, S, K, T, r, b, vol, flag)) for i in range(N_repl)]
#print("Mean_convergency_check", np.mean(test_B))
pass
## SP500 
K1,T1,r1,b1,vol1,flag1 = StrikeArray_SP[6], 3,r_sp[9], r_sp[9] - divYield_sp[9], volatility_sp[9,6], "c"
CALL_BS_formula_SP = GBS(spot_sp,K1,T1,r1,b1,vol1,flag1)
CALL_BS_MC_SP = MC_BS(N, spot_sp, K1, T1, r1, b1, vol1, flag1)
test_S = []
[test_S.append(MC_BS(N, spot_sp, K1, T1, r1, b1, vol1, flag1)) for i in range(N_repl)]
print("Mean_convergency_check", np.mean(test_S))
print(CALL_BS_formula_SP)
pass
######### PUT ###############
## BS
### BLOOMBERG
K,T,r,b,vol,flag = StrikeArray_BMB[9],3,r[-1], r[-1] - q[-1],sigma_NR[-1,9], "p"
PUT_BS_formula_BMB= GBS(S,K,T,r,b,vol,flag)
print("BS", PUT_BS_formula_BMB)
test_B = []

PUT_BS_MC = MC_BS(N, S, K, T, r, b, vol, flag)
[test_B.append(MC_BS(N, S, K, T, r, b, vol, flag)) for i in range(N_repl)]
print("Mean_convergency_check_PUT", np.mean(test_B))

## SP500
K1,T1,r1,b1,vol1,flag1 = StrikeArray_SP[6], 3,r_sp[9], r_sp[9] - divYield_sp[9], volatility_sp[9,6], "p"
PUT_BS_formula_SP = GBS(spot_sp,K1,T1,r1,b1,vol1,flag1)
PUT_BS_MC_SP = MC_BS(N, spot_sp, K1, T1, r1, b1, vol1, flag1)
test_S = []
[test_S.append(MC_BS(N, spot_sp, K1, T1, r1, b1, vol1, flag1)) for i in range(N_repl)]
print(PUT_BS_formula_SP)
print("Mean_convergency_check_PUT", np.mean(test_S))

