# Financial-Engineering
Certificates are Structured Financial products available on the secondary market issued by financial institutions. These are composed by a Fixed Income component plus a Derivative instrument (an Option strategy). The most challenging element to be priced is/are the Options, and in this repositories I posted some pricing techniques used. Among the pricing approaches it's possible to use: 
- Black-Scholes-Merton (Plain Vanilla Options)
- Heston semi-closed formula (with 5 optimal parameters tuned using a Nonlinear Least Squares problem [Nelder-Mead + MSE])
- MonteCarlo simulations for Exotic Options (Euler-Maruyama/Ito's Lemma, Heston dynamics)
- CRR for Barrier options with discrete observation.

I learnt these techniques during an internship in BPER Bank. In this repository I only post the key pricing techniques of the options (not the issued Certificates), leaving out data retrievements (implied dividends, implied volatilities, strip of interest rates curves etc.)
