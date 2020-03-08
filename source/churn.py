# Importing libraries
import warnings

# Input data files are available in the "../input/" directory.
import matplotlib.pyplot as plt  # visualization
import numpy as np  # linear algebra
import pandas as pd

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, MinMaxScaler, VectorAssembler
from pyspark.shell import sqlContext

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import UserDefinedFunction

warnings.filterwarnings("ignore")
import plotly.offline as py  # visualization
py.init_notebook_mode(connected=True)  # visualization
from IPython.core.display import display

if __name__ == "__main__":
    spark = SparkSession.builder.appName("RandomForestClassifierExample").getOrCreate()

    # $example on$
    # Load and parse the data file, converting it to a DataFrame.
    data = spark.read.format("csv").option("header", "true").load("../data/dataset.csv")
    display(data)

    #telcom = pd.read_csv(r"../data/dataset.csv")

    telcom = data.toPandas()
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
    telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ", 999999999999999999)
    tel = telcom[telcom["TotalCharges"] == 999999999999999999]
    tel.head()
    # Dropping null values from total charges column which contain .15% missing data
    telcom = telcom[telcom["TotalCharges"] != 999999999999999999]
    telcom = telcom.reset_index()[telcom.columns]

    # convert to float type
    telcom["TotalCharges"] = telcom["TotalCharges"].astype(float)
    telcom["tenure"] = telcom["tenure"].astype(float)

    # replace 'No internet service' to No for the following columns
    replace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']

    for i in replace_cols:
        telcom[i] = telcom[i].replace({'No internet service': 'No'})

    # Separating churn and non churn customers
    churn = telcom[telcom["Churn"] == "Yes"]
    not_churn = telcom[telcom["Churn"] == "No"]

    # Separating catagorical and numerical columns
    Id_col = ['customerID']
    target_col = ["Churn"]
    cat_cols = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
    cat_cols = [x for x in cat_cols if x not in target_col]
    num_cols = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]
    # Binary columns with 2 values
    bin_cols = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
    # Columns more than 2 values
    multi_cols = [i for i in cat_cols if i not in bin_cols]

    # labels
    lab = telcom["Churn"].value_counts().keys().tolist()
    # values
    val = telcom["Churn"].value_counts().values.tolist()

    telcom = pd.get_dummies(telcom, columns=multi_cols)

    dfs = sqlContext.createDataFrame(telcom)

    cat_cols = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
    cat_cols = [x for x in cat_cols if x not in target_col]

    indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in cat_cols]
    assembler = [VectorAssembler(inputCols=[i], outputCol=i + "_index") for i in num_cols]
    normalizers = [MinMaxScaler(inputCol=column + "_index", outputCol=column + "scaled") for column in num_cols]

    pipeline = Pipeline(stages= indexers+assembler+normalizers)

    dfr = pipeline.fit(dfs).transform(dfs)
    #dfr.drop("customerID")
    #for i in cat_cols + num_cols:
    #       dfs = dfs.drop(i).withColumnRenamed(i  +"_index", i)
    dfs.show()
    #encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
    #encoded = encoder.transform(indexed)
    #encoded.show()


