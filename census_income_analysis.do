******************************************************
* Question 9 – Income Distribution in China (2005)
* Do-file: census_income_analysis.do

******************************************************

version 18.0
clear all
set more off

* --------- PATHS (adjust if needed) ---------
global DATA "/mnt/data"
global OUT  "/mnt/data/output"
cap mkdir "$OUT"

log using "$OUT/run.log", replace text

******************************************************
* (a) Prepare data
******************************************************
use "$DATA/census2005_ps2.dta", clear

* Keep adults age 18–65
keep if age >= 18 & age <= 65

* Drop placeholder income value 99999
drop if income == 99999

* Indicator for zero income
gen byte zero_income = (income == 0)

* Share of zero income in adults 18–65
quietly summarize zero_income
display "Share of zero income (age 18–65): " %6.4f r(mean)

******************************************************
* (b) Income distribution checks
******************************************************
* Descriptive stats
summarize income, detail

* Histogram with normal density
histogram income, normal name(h_income, replace) ///
    title("Income Distribution (Adults 18–65)") xtitle("income")
graph export "$OUT/hist_income.png", replace

* Kernel density with normal overlay
kdensity income, normal name(k_income, replace) ///
    title("Kernel Density of Income")

graph export "$OUT/kdensity_income.png", replace

* Log-income (add 1 to allow income=0)
gen double logincome = log(income + 1)

histogram logincome, normal name(h_lincome, replace) ///
    title("Log Income Distribution") xtitle("log(income+1)")
graph export "$OUT/hist_logincome.png", replace

kdensity logincome, normal name(k_lincome, replace) ///
    title("Kernel Density of Log Income")
graph export "$OUT/kdensity_logincome.png", replace

******************************************************
* (c) Probability with assumed log-normal
*    Assume log(income) ~ N(mu=6, sigma=1)
*    Compute P(income > 1500) = P(log(income) > ln(1500))
******************************************************
local mu = 6
local sigma = 1
local y = 1500
local z = (ln(`y') - `mu')/`sigma'

display "log(1500) = " %6.3f ln(`y')
display "Z = (ln(1500) - 6) / 1 = " %6.3f `z'
display "P(income > 1500) = 1 - Phi(Z) = " %6.4f (1 - normal(`z'))

******************************************************
* (d) Province-level merge & plots
******************************************************
* 1) Import cityvar.csv and save as .dta
import delimited "$DATA/cityvar.csv", clear
save "$DATA/cityvar.dta", replace

* 2) Re-open census, keep adult sample, prepare vars
use "$DATA/census2005_ps2.dta", clear
keep if age >= 18 & age <= 65
drop if income == 99999

* Indicator of zero income BEFORE creating nonzero income
gen byte zero_income = (income == 0)

* Mean income excluding zero incomes
gen double income_nz = cond(income > 0, income, .)

* Collapse to province level: mean non-zero income & share zero-income
collapse (mean) income_nz zero_income, by(province_code)
rename income_nz mean_income_nonzero

* 3) Merge with province-level GDP & unemployment
merge 1:1 province_code using "$DATA/cityvar.dta"

* Keep matched observations only (optional but recommended)
keep if _merge == 3
drop _merge

* 4) Scatterplots
twoway scatter mean_income_nonzero GDP, ///
    title("Avg Income (non-zero) vs GDP by Province") ///
    ytitle("Mean income (non-zero)") xtitle("GDP")
graph export "$OUT/scatter_income_GDP.png", replace

twoway scatter zero_income unemployment_rate, ///
    title("Zero-Income Share vs Unemployment Rate") ///
    ytitle("Share zero income") xtitle("Unemployment rate")
graph export "$OUT/scatter_zero_unemp.png", replace

log close

******************************************************
* End of do-file
******************************************************
