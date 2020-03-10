# Importing libraries
import warnings

from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import StringIndexer, MinMaxScaler, VectorAssembler
from pyspark.ml.feature import ChiSqSelector
from pyspark.shell import sqlContext
from pyspark.sql import SparkSession
warnings.filterwarnings("ignore")
import plotly.offline as py  # visualization
py.init_notebook_mode(connected=True)  # visualization
import numpy as np  # linear algebra

if __name__ == "__main__":

    spark = SparkSession.builder.appName("RandomForestClassifierExample").getOrCreate()
    data = spark.read.format("csv").option("header", "true").load("../data/dataset.csv")

    def evaluateLr(rfTransformed, evaluator,i):
        accuracy = evaluator.evaluate(rfTransformed)
        return np.array([i, accuracy])

    telcom = data.toPandas()

    # Replacing spaces with null values in total charges column
    telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ", 999999)

    # Dropping null values from total charges column which contain .15% missing data
    telcom = telcom[telcom["TotalCharges"] != 999999]
    telcom = telcom.reset_index()[telcom.columns]

    # convert to float type
    telcom["TotalCharges"] = telcom["TotalCharges"].astype(float)
    telcom["tenure"] = telcom["tenure"].astype(float)
    telcom["MonthlyCharges"] = telcom["MonthlyCharges"].astype(float)

    # replace 'No internet service' to No for the following columns
    replace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']

    for i in replace_cols:
        telcom[i] = telcom[i].replace({'No internet service': 'No'})

    # Separating churn and non churn customers
    churn = telcom[telcom["Churn"] == "Yes"]
    not_churn = telcom[telcom["Churn"] == "No"]

    # Separating categorical and numerical columns
    Id_col = ['customerID']
    target_col = ["Churn"]
    cat_cols = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
    cat_cols = [x for x in cat_cols if x not in target_col]
    num_cols = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]
    bin_cols = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
    multi_cols = [i for i in cat_cols if i not in bin_cols]
    telcom = pd.get_dummies(telcom, columns=multi_cols)

    dfs = sqlContext.createDataFrame(telcom)
    cat_cols = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
    cat_cols = [x for x in cat_cols if x not in target_col]
    cat_cols_index = []
    for i in cat_cols:
        cat_cols_index.append(i + "_index")

    labelindexers = [StringIndexer(inputCol="Churn", outputCol="label")]
    indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in cat_cols]
    assembler = [VectorAssembler(inputCols=num_cols + cat_cols_index, outputCol=i + "_indexe") for i in num_cols + cat_cols]
    normalizers = [MinMaxScaler(inputCol=column + "_indexe", outputCol=column + "scaled") for column in cat_cols +num_cols]

    featureCols = []
    for i in num_cols +cat_cols:
        featureCols.append(i + "scaled")

    featureAssembler = [VectorAssembler(inputCols=featureCols, outputCol= "features") ]

    pipeline = Pipeline(stages=indexers+assembler+normalizers+featureAssembler+labelindexers)

    dataForUse, dataOther = dfs.randomSplit([0.5, 0.5])

    fdata = pipeline.fit(dataForUse).transform(dataForUse)
    fdata.cache()

    selectorData, trainingData = fdata.randomSplit([0.7, 0.3])

    # Train a  model.
    lr = LogisticRegression(featuresCol='selectedFeatures', labelCol='label', maxIter=10)
    rf = RandomForestClassifier(labelCol="label", featuresCol="selectedFeatures", numTrees=50)

    new_array = np.array(['  ' ,'  '])

    for i in range(1, len(featureCols)+1):
        print("Number of features affected : ", i)
        selector = ChiSqSelector(numTopFeatures=i, featuresCol="features", outputCol="selectedFeatures", labelCol="label")

        result = selector.fit(selectorData).transform(selectorData)

        trainSelected, testSelected = result.randomSplit([0.7, 0.3])
        rfModel = rf.fit(trainSelected)

        rfTransformed = rfModel.transform(testSelected)
        evaluator = BinaryClassificationEvaluator()
        new_array = np.vstack([evaluateLr(rfTransformed, evaluator, i), new_array])

    print(new_array)

spark.stop()
