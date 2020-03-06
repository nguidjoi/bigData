# Importing libraries
import warnings

# Input data files are available in the "../input/" directory.
import matplotlib.pyplot as plt  # visualization
import numpy as np  # linear algebra
import pandas as pd

warnings.filterwarnings("ignore")
import plotly.offline as py  # visualization
py.init_notebook_mode(connected=True)  # visualization

if __name__ == "__main__":

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


    def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}%".format(pct, absolute)


    def churnPlot():
        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
        wedges, texts, autotexts = ax.pie(val, autopct=lambda pct: func(pct, val), textprops=dict(color="w"))
        ax.legend(wedges, lab, title="Légende", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.setp(autotexts, size=8, weight="bold")
        ax.set_title("Churn des clients.")
        plt.show()


    def plot(column, dataF, title):
        val = dataF[column].value_counts().values.tolist()
        lab = dataF[column].value_counts().keys().tolist()
        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
        wedges, texts, autotexts = ax.pie(val, autopct=lambda pct: func(pct, val), textprops=dict(color="w"))
        ax.legend(wedges, lab, title="Légende", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.setp(autotexts, size=8, weight="bold")
        ax.set_title(title)
        plt.show()


    def showCategoricalFeature():
        for i in cat_cols:
            plot(i, churn, "Churn par " + i)
            plot(i, not_churn, "Non Churn par " + i)


    def histogram(firstData, secondData, column):
        n_bins = 10
        fig1, ax1 = plt.subplots()
        labels = ["churn", "no churn"]
        x_multi = [firstData[column], secondData[column]]
        ax1.hist(x_multi, n_bins, histtype='bar', label=labels)
        ax1.legend(prop={'size': 10})
        ax1.set_title('Variation du churn et du no churn par ' + column)
        fig1.tight_layout()
        plt.show()


    def showNumericalFeature():
        for i in num_cols:
            histogram(churn, not_churn, i)


    churnPlot()
    showNumericalFeature()
    showCategoricalFeature()
