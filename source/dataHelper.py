
import pandas as pd


if __name__ == "__main__":
    def verify(data):
        print("\n Features : ", data.columns.tolist())
        print("Null values :  ", data.isnull().sum())
        print("Unique values :  \n ", data.nunique())


    churnData = pd.read_csv(r"../data/dataset.csv")
    # verify(churnData)
    churnData.head()
