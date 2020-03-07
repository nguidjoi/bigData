# Input data files are available in the "../input/" directory.

import warnings

import pandas as pd

warnings.filterwarnings("ignore")
import plotly.offline as py  # visualization
import plotly.graph_objs as go  # visualization

if __name__ == "__main__":
    telcom = pd.read_csv(r"../data/dataset.csv")
    telcom.head()
    lab = telcom["Churn"].value_counts().keys().tolist()
    val = telcom["Churn"].value_counts().values.tolist()

