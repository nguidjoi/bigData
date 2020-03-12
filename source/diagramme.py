# Importing libraries
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from pyspark.shell import sqlContext
warnings.filterwarnings("ignore")
import plotly.offline as py
py.init_notebook_mode(connected=True)

def displayConfusionMatrix(df_cm):
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    plt.show()

def displayCorrelationMatrix(pandasData, num_cols):
    dps = pd.DataFrame(pandasData, columns=num_cols)
    axs = scatter_matrix(dps, alpha=0.2, figsize=(10, 10));
    n = len(dps.columns)
    for i in range(n):
        v = axs[i, 0]
        v.yaxis.label.set_rotation(0)
        v.yaxis.label.set_ha('right')
        v.set_yticks(())
        h = axs[n - 1, i]
        h.xaxis.label.set_rotation(90)
        h.set_xticks(())
    dps.corr()
    plt.show()


if __name__ == "__main__":

    pandasData = pd.read_csv(r"../data/dataset.csv")
    # first few rows
    pandasData.head()

    print("Rows     : ", pandasData.shape[0])
    print("Columns  : ", pandasData.shape[1])
    print("\nFeatures : \n", pandasData.columns.tolist())
    print("\nMissing values :  ", pandasData.isnull().sum().values.sum())
    print("\nUnique values :  \n", pandasData.nunique())

    # Replacing spaces with null values in total charges column
    pandasData['TotalCharges'] = pandasData["TotalCharges"].replace(" ", np.nan)

    # Dropping null values from total charges column which contain 11 missing rows in data
    pandasData = pandasData[pandasData["TotalCharges"].notnull()]
    pandasData = pandasData.reset_index()[pandasData.columns]

    # convert to float type
    pandasData["TotalCharges"] = pandasData["TotalCharges"].astype(float)

    # replace 'No internet service' to No for the following columns
    replace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    for i in replace_cols:
        pandasData[i] = pandasData[i].replace({'No internet service': 'No'})

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


    pandasData["tenure_group"] = pandasData.apply(lambda data: tenure_lab(data), axis=1)

    # Separating churn and non churn customers
    churn = pandasData[pandasData["Churn"] == "Yes"]
    not_churn = pandasData[pandasData["Churn"] == "No"]

    # Separating catagorical and numerical columns
    Id_col = ['customerID']
    target_col = ["Churn"]
    cat_cols = pandasData.nunique()[pandasData.nunique() < 6].keys().tolist()
    cat_cols = [x for x in cat_cols if x not in target_col]
    num_cols = [x for x in pandasData.columns if x not in cat_cols + target_col + Id_col]

    # labels
    lab = pandasData["Churn"].value_counts().keys().tolist()
    # values
    val = pandasData["Churn"].value_counts().values.tolist()
    spark_df = sqlContext.createDataFrame(pandasData)
    spark_df.show
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


    displayCorrelationMatrix()

    def showNumericalFeature():
        for i in num_cols:
            histogram(churn, not_churn, i)


    churnPlot()
    showNumericalFeature()
    showCategoricalFeature()
