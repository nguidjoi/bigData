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
    trace = go.Pie(labels=lab, values=val, /
            marker = dict(colors=['royalblue', 'lime'], line=dict(color="white", width=1.3)),
                     rotation = 90, hoverinfo = "label+value+text", hole = .5)
    layout = go.Layout(
        dict(title="Customer attrition in data", plot_bgcolor="rgb(243,243,243)", paper_bgcolor="rgb(243,243,243)", ))
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)
