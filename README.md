# ML-Assignment

## Introduction
The goal of this assignment is to apply your knowledge about linear regression to a real-world problem. Your
task for the next two weeks will be to analyze the given data using the techniques we have learned up to
now.

![XKCD](./media/XKCD.png)

*Figure 0: Machine Learning explained by xkcd.com*

Injection molding (Spritzgiessen) is a highly complex process. Environmental conditions, varying characteristics of the input materials, and internal machine parameters and conditions have a direct impact on the
produced components. Often the machine operator has to tune machine settings for each new part in order
to achieve an optimal performance and quality. The institutes ICOM and IWK are currently working on a
joint project with the goal to develop an automated quality assurance system for injection molding machines.
For a set of produced parts, internal machine measurements and parameters, as well as the resulting mass of
the part, have been logged. Your task will be to estimate the mass of a part, given the machine measurements.
With that, the optimal amount of raw material for new parts can be found. The produced part is an ice
scraper, as shown in Figure 1.
![Ice scraper](./media/icescraper.jpg)
*Figure 1: Ice scraper*

The injection molding machine with its components is shown in figure 2
![injection molding machine](./media/injectionmoldingmachine.PNG)
*Figure 2: Injection molding machine*

## Description of Dataset
The dataset consists of two CSV files: InjectionMoldingData_Train.csv, which contains all training samples, and InjectionMoldingData_Test.csv, which contains all test samples. Both files contain 9 columns:
8 predictors and the response. There are 150 training samples and 82 test samples available. All variables are
described in Table 1. A drawing of a typical injection molding machine and the labels of each measurement
is shown in Figure 2.

Name | Description
---- | -----------
PowTotAct_Min | Total power consumption of the machine
Inj1PosVolAct_Var | Position of the screw
Inj1PrsAct_meanOfInjPhase | Melt pressure on screw
Inj1HopTmpAct_1stPCscore | Temperature of the flange
Inj1HtgEd3Act_1stPCscore | Cylinder heating
ClpFceAct_1stPCscore | Clamping force
ClpPosAct_1stPCscore | Clamp position
OilTmp1Act_1stPCscore | Oil temperature
mass | Mass of the produced part

*Table 1: Description of predictors and response of the injection molding dataset.*


## Questions
1. Analyze the training data. Is there a variable which is highly correlated to another variable? List all
variables with correlation coefficients ≥ 0.9.
2. Assume you can only choose one feature to predict the mass as well as possible. Which variable do you
select? Explain why you select this variable and show the relevant numbers (p-value and R2
-value).
3. Build a linear regression model which uses as many input variables as required. Keep in mind that
each sensor costs money, so remove variables which are not needed from the model. List the selected
variables and the relevant numbers for selecting them (p-value and R2
-value).
4. Use the selected model to predict the mass on the test data. Compare the training MSE to the test
MSE. How does your model perform on the test data?
5. Add higher-order terms, such as quadratic terms or interaction terms to improve the model. Judge the
model quality (R2 values). Again, compare the training MSE to the test MSE.

## Conditions
1. The next two weeks (week 5 and 6 of the semester), you independently work on this assignment.
Attendance is not required for these two weeks. The lab 6.004 is available to you at the normal hours,
if you need a computer with Python, Matlab, and RStudio installed, or if you need any help with the
assignment.
2. You write a short report which answers the questions above and summarizes your findings. The report
should be 1 – 2 pages A4 and written in German or English.
3. The report is due on Friday, 25 October 2019 at 17:00. Please also hand in all code written for this
assignment. The report in PDF form and all code have to be sent to sjecklin@hsr.ch. Add the subject
(Betreff) ML-Assignment to the email.
4. Everybody hands in their own report. Collaboration and discussion between students, however, is
allowed and encouraged.

# Report
## Setup
First, we prepare the program to read in the training data and the test data. I used panda for this
and i also imported numpy for later use

```
import numpy as np
import pandas as pd


def main():
    training_data_input = "../data/InjectionMolding_Train.csv"
    test_data_input = "../data/InjectionMolding_Test.csv"
    training_data = pd.read_csv(training_data_input, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    test_data = pd.read_csv(test_data_input, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9])


if __name__ == '__main__':
    main()
```

## 1. Correlation between variables
For the correlation between variables, i decided to use the pandas correlation function

```
def correlate_data(data):
    corr = data.corr()
    return corr


def plot_correlation_matrix(correlation_data):
    plt.figure(figsize=(8, 8))
    plt.imshow(correlation_data, interpolation='none', aspect='auto')
    plt.title('Correlation Matrix', fontsize=18)
    plt.colorbar()
    plt.show()


# Reduce the redundant pairs and the correlation from a predictor with itself
def get_redundant_pairs(data):
    pairs_to_drop = set()
    cols = data.columns
    for i in range(0, data.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


# Get the top five correlations between predictors, redundancy excluded
def get_top_abs_correlations(data, n=5):
    au_corr = data.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(data)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]
```

And in the main() function, add:

```
correlation_data = correlate_data(training_data)
print(get_top_abs_correlations(correlation_data))
plot_correlation_matrix(correlation_data)
```
The results are a listing of the top five correlations and a plot of the correlation matrix.
```
PowTotAct_Min      Inj1PosVolAct_Var           0.995158
                   OilTmp1Act_1stPCscore       0.964045
Inj1PosVolAct_Var  OilTmp1Act_1stPCscore       0.944855
                   Inj1HopTmpAct_1stPCscore    0.906364
PowTotAct_Min      Inj1HopTmpAct_1stPCscore    0.870516
dtype: float64 
```

![Plot](./media/myplot.png)

*Figure 3: Plotted Correlation Matrix*

We have multiple pairs that correlate with more than 0.9: 
```
PowTotAct_Min      Inj1PosVolAct_Var           0.995158
                   OilTmp1Act_1stPCscore       0.964045
Inj1PosVolAct_Var  OilTmp1Act_1stPCscore       0.944855
                   Inj1HopTmpAct_1stPCscore    0.906364
```

Total power consumption of the machine (PowTotAct_Min) correlates with
Position of the screw (Inj1PosVolAct_Var) and Oil temperature (OilTmp1Act_1stPCscore)

The Position of the screw (Inj1PosVolAct_Var) also correlates with the Oil temperature (OilTmp1Act_1stPCscore) 
and with the Temperature of the flange (Inj1HopTmpAct_1stPCscore)

## 2. Which predictor to choose?
For this part of the assignment, i used the statsmodels.formula.api module to directly print a result set with
 all relevant information. the method works as follows:
```
def statsmodel_regression(training_data, formula: str):
    model = sm.ols(formula, training_data)
    result = model.fit()
    parameters = result.summary()
    return parameters
```
in the main() i add the following lines
```
formula = "mass ~ PowTotAct_Min " \
              "+ Inj1PosVolAct_Var" \
              "+ Inj1PrsAct_meanOfInjPhase" \
              "+ Inj1HopTmpAct_1stPCscore" \
              "+ Inj1HtgEd3Act_1stPCscore" \
              "+ ClpFceAct_1stPCscore" \
              "+ ClpPosAct_1stPCscore" \
              "+ OilTmp1Act_1stPCscore"
    parameters = statsmodel_regression(training_data, formula)
    print(parameters)
```
The resulting table shows all relevant data to choose the most significant predictor.
```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   mass   R-squared:                       0.987
Model:                            OLS   Adj. R-squared:                  0.986
Method:                 Least Squares   F-statistic:                     1312.
Date:                Thu, 17 Oct 2019   Prob (F-statistic):          2.55e-128
Time:                        17:55:09   Log-Likelihood:                 459.93
No. Observations:                 150   AIC:                            -901.9
Df Residuals:                     141   BIC:                            -874.8
Df Model:                           8                                         
Covariance Type:            nonrobust                                         
=============================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------------
Intercept                    30.3054      0.356     85.160      0.000      29.602      31.009
PowTotAct_Min              8.399e-07   1.06e-06      0.789      0.431   -1.26e-06    2.94e-06
Inj1PosVolAct_Var             0.0052      0.001      6.772      0.000       0.004       0.007
Inj1PrsAct_meanOfInjPhase     0.0044      0.001      7.827      0.000       0.003       0.006
Inj1HopTmpAct_1stPCscore     -0.0001      0.000     -0.309      0.758      -0.001       0.001
Inj1HtgEd3Act_1stPCscore      0.0005   4.74e-05     10.508      0.000       0.000       0.001
ClpFceAct_1stPCscore         -0.0004   4.19e-05     -9.012      0.000      -0.000      -0.000
ClpPosAct_1stPCscore         -0.0002      0.000     -1.162      0.247      -0.001       0.000
OilTmp1Act_1stPCscore         0.0014      0.000     10.888      0.000       0.001       0.002
```
The R-Squared value shows, that the model explains a large portion of the variance in the response variable.
The P>|t| values show significance for the model and in general a value lower than 0.005 is preferable.

This means, that the position of the screw (Inj1PosVolAct_Var), the melt pressure of the screw 
(Inj1PrsAct_meanOfInjPhase), the cilinder heating (Inj1HtgEd3Act_1stPCscore), 
the clamp force (ClpFceAct_1stPCscore) and the oil temperature (OilTmp1Act_1stPCscore) are all
viable predictors for the response. 

Since the position of the screw and the oil temperature correlate with other predictors, i discard 
them as predictors. I decided to choose the melt pressure of the screw, since it has a higher coefficient
than the other two, meaning, that if i change it by one unit, in average the response changes more, provided
everything else is constant.

## 3. Model with only relevant predictors
For this model, i reuse the statsmodel_regression method with a backward selection process, meaning that in every
iteration i remove the predictor with the highest value and check, if the R-squared and standard error of the 
coefficient gets better.
```
 # 3. Reduction to relevant predictors
    formula = "mass ~ PowTotAct_Min " \
              "+ Inj1PosVolAct_Var" \
              "+ Inj1PrsAct_meanOfInjPhase" \
              "+ Inj1HtgEd3Act_1stPCscore" \
              "+ ClpFceAct_1stPCscore" \
              "+ ClpPosAct_1stPCscore" \
              "+ OilTmp1Act_1stPCscore"
    parameters = statsmodel_regression(training_data, formula)
    print(parameters)
    # exclusion of the Inj1HopTmpAct_1stPCscore leads to a .003  better std err of the intercept

    formula = "mass ~  Inj1PosVolAct_Var" \
              "+ Inj1PrsAct_meanOfInjPhase" \
              "+ Inj1HtgEd3Act_1stPCscore" \
              "+ ClpFceAct_1stPCscore" \
              "+ ClpPosAct_1stPCscore" \
              "+ OilTmp1Act_1stPCscore"
    parameters = statsmodel_regression(training_data, formula)
    print(parameters)
    # exclusion of PowTotAct_Min has no real impact on the std error

    formula = "mass ~  Inj1PosVolAct_Var" \
              "+ Inj1PrsAct_meanOfInjPhase" \
              "+ Inj1HtgEd3Act_1stPCscore" \
              "+ ClpFceAct_1stPCscore" \
              "+ OilTmp1Act_1stPCscore"
    parameters = statsmodel_regression(training_data, formula)
    print(parameters)
    # exclusion of ClpPosAct_1stPCscore reduces the std error by .001
```
For testing, i excluded all variables each in a separate iteration to check, if there is some significant 
change. The exclusion of the Inj1PosVolAct_Var reduced the standard error of 
the coefficient by .285

## 4. Let the tests roll in 
Now its time to evaluate the system with the test-data