import numpy as np
from HESTON_MODEL_INPUTS_BLOOMBERG import r, q,S, StrikeArray_BMB, sigma_NR,GBS
from HESTON_MODEL_INPUTS_SP500 import r_sp, divYield_sp, StrikeArray_SP, volatility_sp,spot_sp
import HESTON_CLASSES as H_CS
from MAIN import Heston_pricer

def MC_HE(repl, intervals,params,optiontype):
    kappa, theta,v0, sigma,rho,lamda = params[0][0], params[0][1], params[0][2],params[0][3],params[0][4], 0
    S, K, T, r, q = params[1][0], params[1][1], params[1][2],params[1][3],params[1][4]

    sim = np.ones((intervals, repl)) * S
    var = np.ones((intervals, repl)) * v0
    dt = T / intervals

    Zv = np.random.randn(intervals, repl)
    Zs = rho * Zv + np.sqrt(1 - rho ** 2) * np.random.randn(intervals, repl)
    F=0
    for i in range(1, intervals):
        var[i,:] = np.abs(var[i - 1,:] + kappa * (theta - var[i - 1,:]) * dt + sigma * np.sqrt(var[i - 1,:]) * np.sqrt(dt) * Zv[i,:])
        sim[i,:] = sim[i - 1,:] * np.exp((r - q - 0.5 * var[i - 1,:]) * dt + np.sqrt(var[i - 1,:]) * np.sqrt(dt) * Zs[i,:])
    discount = np.exp(-r * T)
    payoff = np.maximum(0, sim[-1,:] - K) if optiontype.lower().startswith("c") else np.maximum(0,K - sim[-1,:])
    option_price = np.mean(payoff) * discount
    return option_price

## OPTIMAL PARAMETERS FROM min SUM SQUARE ERROR
Theta_B = [1.5213994,0.03550201,0.01327494,0.32858323,-0.64150853]
Theta_S = [0.51196095,0.03452276,0.02226145,0.18800531,-0.90550424]
settings_B = [S, StrikeArray_BMB[9], 3, r[-1], q[-1], sigma_NR[-1,9]]
settings_S = [S, StrikeArray_SP[6], 3, r_sp[9], divYield_sp[9], volatility_sp[9,6]]

## PARAMS & SETTINGS BLOOMBERG
S_Bl,K_Bl,T_Bl,r_Bl,q_Bl,vol_Bl = S,StrikeArray_BMB[9],3,r[-1],q[-1],sigma_NR[-1,9]
kappa_B,theta_B,v0_B,sigma_B,rho_B = Theta_B[0], Theta_B[1], Theta_B[2], Theta_B[3], Theta_B[4]
C_Carry_B= r_Bl-q_Bl

## PARAMS & SETTINGS SP500
S_SP,K_SP,T_SP,r_SP,q_SP,vol_SP = S,StrikeArray_SP[6], 3,r_sp[9],divYield_sp[9], volatility_sp[9,6]
kappa_S,theta_S,v0_S,sigma_S,rho_S = Theta_S[0], Theta_S[1], Theta_S[2], Theta_S[3], Theta_S[4]
C_Carry_SP = r_sp[9] - divYield_sp[9]

optiontype = "c"
GBS_call = GBS(S_Bl,K_Bl,T_Bl,r_Bl,C_Carry_B,vol_Bl,optiontype)
print("GBS c bmb",GBS_call)
Hparam = H_CS.HParam(kappa_B,theta_B,v0_B,sigma_B,rho_B,0)
Settings = H_CS.OPset(S_Bl,K_Bl,T_Bl,r_Bl,q_Bl,optiontype,1)
CALL_HE_formula_BMB = Heston_pricer(Hparam,Settings)
print("HESTON_c bmb: ",CALL_HE_formula_BMB)

optiontype = "p"
GBS_put = GBS(S_Bl,K_Bl,T_Bl,r_Bl,C_Carry_B,vol_Bl,optiontype)
print("BS_p bmb ",GBS_call)
Hparam = H_CS.HParam(kappa_B,theta_B,v0_B,sigma_B,rho_B,0)
Settings = H_CS.OPset(S_Bl,K_Bl,T_Bl,r_Bl,q_Bl,optiontype,1)
PUT_HE_formula_BMB = Heston_pricer(Hparam,Settings)
print("HESTON_p bmb: ",PUT_HE_formula_BMB)
print("################### sp")
optiontype = "c"
GBS_call = GBS(S_SP,K_SP,T_SP,r_SP,C_Carry_SP,vol_SP,optiontype)
print("BS_c sp: ",GBS_call)
Hparam = H_CS.HParam(kappa_S,theta_S,v0_S,sigma_S,rho_S,0)
Settings = H_CS.OPset(S_SP,K_SP,T_SP,r_SP,q_SP,optiontype,1)
CALL_HE_formula_SP500 = Heston_pricer(Hparam,Settings)
print("HESTON_c sp: ",CALL_HE_formula_SP500)

optiontype = "p"
GBS_put = GBS(S_SP,K_SP,T_SP,r_SP,C_Carry_SP,vol_SP,optiontype)
print("BS_p: ",GBS_put)
Hparam = H_CS.HParam(kappa_S,theta_S,v0_S,sigma_S,rho_S,0)
Settings = H_CS.OPset(S_SP,K_SP,T_SP,r_SP,q_SP,optiontype,1)
PUT_HE_formula_SP500= Heston_pricer(Hparam,Settings)
print("HESTON_p: ",PUT_HE_formula_SP500)

########## CALL ###########
##BLOOMBERG
###### MC #######
option_price = MC_HE(20000,1080,params=[Theta_S,settings_S],optiontype="call")
pass
test_B= []
[test_B.append(MC_HE(20000,1080,params=[Theta_B,settings_B],optiontype="put"))for i in range(100)]
print(np.mean(test_B))