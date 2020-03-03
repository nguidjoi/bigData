# Importing libraries
import warnings

# Input data files are available in the "../input/" directory.
import matplotlib.pyplot as plt  # visualization
import numpy as np  # linear algebra
import pandas as pd
import seaborn as sns  # visualization

warnings.filterwarnings("ignore")
import plotly.offline as py  # visualization

py.init_notebook_mode(connected=True)  # visualization
import plotly.graph_objs as go  # visualization

if __name__ == "__main__":

    sns.set()

    telcom = pd.read_csv(r"../data/dataset.csv")
    # first few rows
    telcom.head()

    print("Rows     : ", telcom.shape[0])
    print("Columns  : ", telcom.shape[1])
    print("\nFeatures : \n", telcom.columns.tolist())
    print("\nMissing values :  ", telcom.isnull().sum().values.sum())
    print("\nUnique values :  \n", telcom.nunique())

    # Data Manipulation

    # Data Manipulation

    # Replacing spaces with null values in total charges column
    telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ", np.nan)

    # Dropping null values from total charges column which contain .15% missing data
    telcom = telcom[telcom["TotalCharges"].notnull()]
    telcom = telcom.reset_index()[telcom.columns]

    # convert to float type
    telcom["TotalCharges"] = telcom["TotalCharges"].astype(float)

    # replace 'No internet service' to No for the following columns
    replace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    for i in replace_cols:
        telcom[i] = telcom[i].replace({'No internet service': 'No'})

    # replace values
    telcom["SeniorCitizen"] = telcom["SeniorCitizen"].replace({1: "Yes", 0: "No"})


    # Tenure to categorical column
    def tenure_lab(telcom):

        if telcom["tenure"] <= 12:
            return "Tenure_0-12"
        elif (telcom["tenure"] > 12) & (telcom["tenure"] <= 24):
            return "Tenure_12-24"
        elif (telcom["tenure"] > 24) & (telcom["tenure"] <= 48):
            return "Tenure_24-48"
        elif (telcom["tenure"] > 48) & (telcom["tenure"] <= 60):
            return "Tenure_48-60"
        elif telcom["tenure"] > 60:
            return "Tenure_gt_60"


    telcom["tenure_group"] = telcom.apply(lambda telcom: tenure_lab(telcom),
                                          axis=1)

    # Separating churn and non churn customers
    churn = telcom[telcom["Churn"] == "Yes"]
    not_churn = telcom[telcom["Churn"] == "No"]

    # Separating catagorical and numerical columns
    Id_col = ['customerID']
    target_col = ["Churn"]
    cat_cols = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
    cat_cols = [x for x in cat_cols if x not in target_col]
    num_cols = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]

    # labels
    lab = telcom["Churn"].value_counts().keys().tolist()
    # values
    val = telcom["Churn"].value_counts().values.tolist()

    trace = go.Pie(labels=lab,
                   values=val,
                   marker=dict(colors=['royalblue', 'lime'],
                               line=dict(color="white",
                                         width=1.3)
                               ),
                   rotation=90,
                   hoverinfo="label+value+text",
                   hole=.5
                   )
    layout = go.Layout(dict(title="Customer attrition in data",
                            plot_bgcolor="rgb(243,243,243)",
                            paper_bgcolor="rgb(243,243,243)",
                            )
                       )

    data = [trace]
    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig)
    plt.show()

    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap(np.arange(3) * 4)
    inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))
    size = 0.3
    day = [1, 2, 3, 4, 5]
    sleeping = [7, 8, 6, 11, 7]
    eating = [2, 3, 4, 3, 2]
    working = [7, 8, 7, 2, 2]
    playing = [8, 5, 7, 8, 13]
    slices = [7, 2, 2, 13]
    activities = ['sleeping', 'eating', 'working', 'playing']
    cols = ['y', 'b', 'r', 'g']
    plt.pie(slices, labels=activities, wedgeprops=dict(edgecolor="white", width=0.4), colors=outer_colors,
            startangle=45, shadow=False, explode=(0, 0, 0, 0), radius=0.8)
    plt.show()
