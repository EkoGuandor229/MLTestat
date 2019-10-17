import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
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
    correlation_data = correlate_data(training_data)
    print(get_top_abs_correlations(correlation_data))
    plot_correlation_matrix(correlation_data)

    # 2. Singular Predictor choice
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

    formula = "mass ~  " \
              "+ Inj1PrsAct_meanOfInjPhase" \
              "+ Inj1HtgEd3Act_1stPCscore" \
              "+ ClpFceAct_1stPCscore" \
              "+ OilTmp1Act_1stPCscore"
    parameters = statsmodel_regression(training_data, formula)
    print(parameters)
    # exclusion of Inj1PosVolAct_Var reduces the std error by .285!

    # 4. Test the model
    formula = "mass ~ Inj1PrsAct_meanOfInjPhase" \
              "+ Inj1HtgEd3Act_1stPCscore" \
              "+ ClpFceAct_1stPCscore" \
              "+ OilTmp1Act_1stPCscore"
    prediction = statsmodel_prediction(training_data, test_data_predictors, formula)
    print(prediction)
    # calculate mse's


def statsmodel_regression(training_data, formula: str):
    model = sm.ols(formula, training_data)
    result = model.fit()
    parameters = result.summary()
    return parameters


def statsmodel_prediction(training_data, test_data, formula: str):
    model = sm.ols(formula, training_data).fit()
    result = model.predict(test_data)
    return result


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
