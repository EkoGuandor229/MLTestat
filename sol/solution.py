import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt






def main():
    training_data_input = "../data/InjectionMolding_Train.csv"
    test_data_input = "../data/InjectionMolding_Test.csv"
    training_data = pd.read_csv(training_data_input, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    test_data = pd.read_csv(test_data_input, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9])

    correlation_data = correlate_data(training_data)
    print(get_top_abs_correlations(correlation_data))
    plot_correlation_matrix(correlation_data)

    # Parameters
    # x1 - PowTotAct_Min
    # x2 - Inj1PosVolAct_Var
    # x3 - Inj1PrsAct_meanOfInjPhase
    # x4 - Inj1HopTmpAct_1stPCscore
    # x5 - Inj1HtgEd3Act_1stPCscore
    # x6 - ClpFceAct_1stPCscore
    # x7 - ClpPosAct_1stPCscore
    # x8 - OilTmp1Act_1stPCscore
    # y - mass


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
