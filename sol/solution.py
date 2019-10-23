import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import numpy as np
from statsmodels.tools.eval_measures import mse


def main():
    training_data_input = "../data/InjectionMolding_Train.csv"
    test_data_input = "../data/InjectionMolding_Test.csv"
    training_data = pd.read_csv(training_data_input, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    test_data_predictors = pd.read_csv(test_data_input, usecols=[0, 1, 2, 3, 4, 5, 6, 7])
    test_data_response = pd.read_csv(test_data_input, usecols=[8])

    # Parameters
    # x1 = training_data.PowTotAct_Min
    # x2 = training_data.Inj1PosVolAct_Var
    # x3 = training_data.Inj1PrsAct_meanOfInjPhase
    # x4 = training_data.Inj1HopTmpAct_1stPCscore
    # x5 = training_data.Inj1HtgEd3Act_1stPCscore
    # x6 = training_data.ClpFceAct_1stPCscore
    # x7 = training_data.ClpPosAct_1stPCscore
    # x8 = training_data.OilTmp1Act_1stPCscore
    # y = training_data.mass

    # 1. Correlation
    print("Correlation")
    correlation_data = correlate_data(training_data)
    print(get_top_abs_correlations(correlation_data))
    plot_correlation_matrix(correlation_data)
    print("")

    # 2. Singular Predictor choice
    print("Analysis of the whole system")
    formula = "mass ~ PowTotAct_Min " \
              "+ Inj1PosVolAct_Var" \
              "+ Inj1PrsAct_meanOfInjPhase" \
              "+ Inj1HopTmpAct_1stPCscore" \
              "+ Inj1HtgEd3Act_1stPCscore" \
              "+ ClpFceAct_1stPCscore" \
              "+ ClpPosAct_1stPCscore" \
              "+ OilTmp1Act_1stPCscore"
    model = sm.ols(formula, training_data).fit()
    print(model.summary())
    print("")
    # in the list, we see that there are five different predictors with a p-value lower than 0.005
    # The next step is to print the list for each predictor so that mass = predictor

    print("Parameters of position of the screw")
    formula = "mass ~ Inj1PosVolAct_Var"
    model = sm.ols(formula, training_data).fit()
    print("{} \n{}".format(model.rsquared, model.pvalues[1]))
    print("")

    print("Parameters of melt pressure of the screw")
    formula = "mass ~ Inj1PrsAct_meanOfInjPhase"
    model = sm.ols(formula, training_data).fit()
    print("{} \n{}".format(model.rsquared, model.pvalues[1]))
    print("")

    print("Parameters of Cylinder heating")
    formula = "mass ~ Inj1HtgEd3Act_1stPCscore"
    model = sm.ols(formula, training_data).fit()
    print("{} \n{}".format(model.rsquared, model.pvalues[1]))
    print("")

    print("Parameters of clamp force")
    formula = "mass ~ ClpFceAct_1stPCscore"
    model = sm.ols(formula, training_data).fit()
    print("{} \n{}".format(model.rsquared, model.pvalues[1]))
    print("")

    print("Parameters of the oil temperature")
    formula = "mass ~ OilTmp1Act_1stPCscore"
    model = sm.ols(formula, training_data).fit()
    print("{} \n{}".format(model.rsquared, model.pvalues[1]))
    print("")

    # 3. Reduction to relevant predictors
    print("Add posistion of the screw to melt pressure")
    formula = "mass ~ Inj1PrsAct_meanOfInjPhase" \
              "+ Inj1PosVolAct_Var"
    model = sm.ols(formula, training_data).fit()
    print("{} \n{}".format(model.rsquared, model.pvalues))
    print("")

    print("Add clamp force")
    formula = "mass ~ Inj1PrsAct_meanOfInjPhase" \
              "+ Inj1PosVolAct_Var" \
              "+ ClpFceAct_1stPCscore"
    model = sm.ols(formula, training_data).fit()
    print("{} \n{}".format(model.rsquared, model.pvalues))
    print("")

    print("Add oil cylinder heating")
    formula = "mass ~ Inj1PrsAct_meanOfInjPhase" \
              "+ Inj1PosVolAct_Var" \
              "+ ClpFceAct_1stPCscore" \
              "+ Inj1HtgEd3Act_1stPCscore"
    model = sm.ols(formula, training_data).fit()
    print("{} \n{}".format(model.rsquared, model.pvalues))
    print("")

    print("Add oil temperature")
    formula = "mass ~ Inj1PrsAct_meanOfInjPhase" \
              "+ Inj1PosVolAct_Var" \
              "+ ClpFceAct_1stPCscore" \
              "+ Inj1HtgEd3Act_1stPCscore" \
              "+ OilTmp1Act_1stPCscore"
    model = sm.ols(formula, training_data).fit()
    print("{} \n{}".format(model.rsquared, model.pvalues))
    print("")

    # calculate mse's
    # Training data Mean Squared Errors
    model = sm.ols(formula, training_data).fit()
    predictions = model.predict(training_data)
    mean_squared_error = (np.mean(np.square(training_data.mass - predictions)))
    print("Training-MSE: " + str(mean_squared_error))

    # Test data mean squared errors
    model = sm.ols(formula, training_data).fit()
    predictions = model.predict(test_data_predictors)
    mean_squared_error = np.mean(np.square(test_data_response.mass - predictions))
    print("Test-MSE: " + str(mean_squared_error))
    print("")

    # Interaction terms
    formula = "mass ~ I(Inj1PrsAct_meanOfInjPhase ** 2) + Inj1PosVolAct_Var"
    model = sm.ols(formula, training_data).fit()
    print("{} \n{}".format(model.rsquared, model.pvalues))
    predictions = model.predict(test_data_predictors)
    mean_squared_error = np.mean(np.square(test_data_response.mass - predictions))
    print("Test-MSE: " + str(mean_squared_error))
    print("")

    formula = "mass ~ I(Inj1PrsAct_meanOfInjPhase ** 2) + Inj1HtgEd3Act_1stPCscore"
    model = sm.ols(formula, training_data).fit()
    print("{} \n{}".format(model.rsquared, model.pvalues))
    predictions = model.predict(test_data_predictors)
    mean_squared_error = np.mean(np.square(test_data_response.mass - predictions))
    print("Test-MSE: " + str(mean_squared_error))
    print("")

    formula = "mass ~ I(Inj1PrsAct_meanOfInjPhase ** 2) + ClpFceAct_1stPCscore"
    model = sm.ols(formula, training_data).fit()
    print("{} \n{}".format(model.rsquared, model.pvalues))
    predictions = model.predict(test_data_predictors)
    mean_squared_error = np.mean(np.square(test_data_response.mass - predictions))
    print("Test-MSE: " + str(mean_squared_error))
    print("")

    formula = "mass ~ I(Inj1PrsAct_meanOfInjPhase ** 2) + OilTmp1Act_1stPCscore"
    model = sm.ols(formula, training_data).fit()
    print("{} \n{}".format(model.rsquared, model.pvalues))
    predictions = model.predict(test_data_predictors)
    mean_squared_error = np.mean(np.square(test_data_response.mass - predictions))
    print("Test-MSE: " + str(mean_squared_error))


def correlate_data(data):
    corr = data.corr()
    return corr


def plot_correlation_matrix(correlation_data):
    plt.figure(figsize=(8, 8))
    plt.imshow(correlation_data, interpolation='none', aspect='auto')
    plt.title('Correlation Matrix', fontsize=18)
    plt.colorbar()
    plt.show()


def get_redundant_pairs(data):
    pairs_to_drop = set()
    cols = data.columns
    for i in range(0, data.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(data, n=5):
    au_corr = data.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(data)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


if __name__ == '__main__':
    main()
