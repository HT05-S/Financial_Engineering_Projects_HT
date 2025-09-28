## Project Report: Continuously Compounded Discount Rate Computation

## 1. Project Overview-
The project is designed towards computing pricing information from coupon-paying US Treasury bonds to estimate the correct discount rate for ZCBs maturing any date in the future. This process is usually known as bootstrapping but we will use modelling techniques to fit a given dataset into a mathematical equation for r(t). 

## 2. Usecase- 
- r(t) fitting is done using two functions i) Model 01 r(t) =𝑎 +𝑏𝑡+𝑐𝑐 𝑡2+𝑑 𝑒𝑓𝑡 +𝑔 ln (𝑡−ℎ) 
                                           ii) Model 02 r(t) = Nelson-Siegel Model 

## 3. Project Lifecycle & Deliverables-
- First, select function r(t). 
- Compute coupon dates for all given bonds. 
- Now, for each coupon date calculate the number of years (t) between settlement date 
and coupon dates i.e. t = coupon date - settlement date.  
- For all coupon dates as well as maturity dates (T) compute discount factor dt = exp[-r(t)t]. 
- Now for each bond calculate fitted bond dirty value and subtract accrued interest from it. 
This will give us a clean price of bond calculated using r(t).  
FittedPrice (fitted clean price) = Fitted bond dirty value − Accrued Interest 
- Finally computing sum of squared error between FittedPrice and MarketPrice.
A key technical result from the fine-tuning process was the determination of an optimal softmax temperature of 1.1315. This parameter was crucial in calibrating the model's confidence scores, yielding more reliable probability distributions for its predictions.

## 4. Results & Discussion-
- The project successfully created a functional tool for computation of Dicount rate or Zero rate from Coupon bearing US Treasuries.
- We observed that the variation of residuals seems to be high in model 2, especially towards the end of the curve, compared to model 1 wherein we see a lower value of residuals comparatively.  
- We believe that NSS performed relatively worse in modelling longer duration bonds compared to Model 1 which could fit the curve better due to having more regression coefficients and better flexibility in fitting the model. 

## 5. Risks-
- Overfitting risk due to excessive model flexibility, especially in Model 01 with multiple parameters.
- Numerical instability in optimization if the initial guesses or parameter bounds are poorly chosen.
- Dependency on data quality — missing or noisy bond price data can distort fitted discount rates.
- Logarithmic term in Model 01 introduces domain restrictions (t > h), requiring additional safeguards during computation.
- The chosen settlement date and day-count convention assumptions may influence results if not consistent with market standards.

## 6. Assumptions-
- Accrued interest uses Actual/Actual i.e. days / 365 fractional year calculation.  
- Considered 02-25-2025 as the settlement date. 
- Model 01 contains a logarithm term ln (t−h) it requires t>h. The optimizer was constrained so h remains smaller than the smallest positive cashflow time.  
- Both models minimized the mean sum of squared errors on clean prices.

## 7. Local Setup-
- Go to Data folder in Continuously_compounded_Discount_rates_from_CouponBearing_USTreasury_Bonds Folder.
- Use dataset present in data folder for analysis.
- Go to Model01 folder in Notebooks folder, run Model01.ipynb file.
- Go to Model02-NS folder in Notebooks folder, run Model02.ipynb file.
