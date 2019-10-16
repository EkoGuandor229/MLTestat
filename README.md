# ML-Assignment

## Introduction
The goal of this assignment is to apply your knowledge about linear regression to a real-world problem. Your
task for the next two weeks will be to analyze the given data using the techniques we have learned up to
now.

Injection molding (Spritzgiessen) is a highly complex process. Environmental conditions, varying characteristics of the input materials, and internal machine parameters and conditions have a direct impact on the
produced components. Often the machine operator has to tune machine settings for each new part in order
to achieve an optimal performance and quality. The institutes ICOM and IWK are currently working on a
joint project with the goal to develop an automated quality assurance system for injection molding machines.
For a set of produced parts, internal machine measurements and parameters, as well as the resulting mass of
the part, have been logged. Your task will be to estimate the mass of a part, given the machine measurements.
With that, the optimal amount of raw material for new parts can be found. The produced part is an ice
scraper, as shown in Figure 1.

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
First, we prepare the program to read in the training data and the test data.

## 1. Correlation between variables
