
import pandas as pd


if __name__ == "__main__":
    def verify(dataSet):
        print("\n Features : \n", dataSet.columns.tolist())
        print("\n Null values :  ", dataSet.isnull().sum())
        print("\n Missing values :  ", dataSet.isnull().sum().values.sum())
        print("\n Unique values :  \n", dataSet.nunique())

    churnData = pd.read_csv(r"../data/dataset.csv")
    verify(churnData)