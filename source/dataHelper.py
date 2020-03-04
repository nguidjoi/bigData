
import pandas as pd


if __name__ == "__main__":
    def verify(dataSet):

        print("\nFeatures : \n", dataSet.columns.tolist())
        print("\nNull values :  ", dataSet.isnull().sum())
        print("\nMissing values :  ", dataSet.isnull().sum().values.sum())
        print("\nUnique values :  \n", dataSet.nunique())

    churnData = pd.read_csv(r"../data/dataset.csv")
    verify(churnData)