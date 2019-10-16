import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def correlate(x):
    plt.matshow(x.corr())


def main():
    training_data_input = "../data/InjectionMolding_Train.csv"
    test_data_input = "../data/InjectionMolding_Test.csv"
    training_data = pd.read_csv(training_data_input, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    test_data = pd.read_csv(test_data_input, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9])

    corr = training_data.corr()
    print(corr)
    f = plt.figure(figsize=(8, 8))
    plt.matshow(corr, fignum=f.number)
    plt.title('Correlation Matrix')
    plt.show()

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


if __name__ == '__main__':
    main()