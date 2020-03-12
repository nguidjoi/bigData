# Importing libraries
import warnings

from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import StringIndexer, MinMaxScaler, VectorAssembler, OneHotEncoderEstimator
from pyspark.ml.feature import ChiSqSelector
from pyspark.shell import sqlContext
from pyspark.sql import SparkSession
warnings.filterwarnings("ignore")
import plotly.offline as py
py.init_notebook_mode(connected=True)
import numpy as np

def evaluatePrediction(rfTransformed, evaluator):
    auroc = evaluator.evaluate(rfTransformed, {evaluator.metricName: "areaUnderROC"})
    auprc = evaluator.evaluate(rfTransformed, {evaluator.metricName: "areaUnderPR"})
    tpr = evaluator.evaluate(rfTransformed, {evaluator.metricName: "truePositiveRate"})
    return np.array([auroc,auprc,tpr])

def evaluateLr(rfTransformed, evaluator, i):
    auroc = evaluator.evaluate(rfTransformed, {evaluator.metricName: "areaUnderROC"})
    auprc = evaluator.evaluate(rfTransformed, {evaluator.metricName: "areaUnderPR"})
    return np.array([i, auroc,auprc])

def initializePipeline(num_cols,cat_cols):

    cat_cols_index = []
    cat_cols_hoted = []
    for i in cat_cols:
        cat_cols_index.append(i + "_index")
        cat_cols_hoted.append(i + "_hoted")

    featureCols = []
    for i in num_cols:
        featureCols.append(i + "scaled")

    for i in cat_cols:
        featureCols.append(i + "_hoted")

    labelindexers = [StringIndexer(inputCol="Churn", outputCol="label")]
    indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in cat_cols]
    oneHotEncoder = [OneHotEncoderEstimator(inputCols=cat_cols_index, outputCols=cat_cols_hoted, dropLast=False)]
    assembler = [VectorAssembler(inputCols=num_cols, outputCol=i + "_indexe") for i in num_cols]
    normalizers = [MinMaxScaler(inputCol=column + "_indexe", outputCol=column + "scaled") for column in num_cols]
    featureAssembler = [VectorAssembler(inputCols=featureCols, outputCol="features")]
    pipeline = Pipeline(stages=indexers + oneHotEncoder + assembler + normalizers + featureAssembler + labelindexers)
    return pipeline


def cleanData(sparkDf):

    pandaDf = sparkDf.toPandas()
    # Replacing spaces with 999999 values in total charges column
    pandaDf['TotalCharges'] = pandaDf["TotalCharges"].replace(" ", 999999)
    # Dropping null values from total charges column which contain 11 rows with missing data
    pandaDf = pandaDf[pandaDf["TotalCharges"] != 999999]
    pandaDf = pandaDf.reset_index()[pandaDf.columns]
    # convert to float type
    pandaDf["TotalCharges"] = pandaDf["TotalCharges"].astype(float)
    pandaDf["tenure"] = pandaDf["tenure"].astype(float)
    pandaDf["MonthlyCharges"] = pandaDf["MonthlyCharges"].astype(float)
    replace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    for i in replace_cols:
        pandaDf[i] = pandaDf[i].replace({'No internet service': 'No'})

    pandaDf["MultipleLines"].replace({'No phone service': 'No'})
    return pandaDf


def loadData(spark):
    data = spark.read.format("csv").option("header", "true").load("../data/dataset.csv").drop(
        "customerID").drop_duplicates()
    return data


def getAllMeasure(rf,selectorData,featureCols,):
    measure = np.array(['  ', '  ', '  '])
    for i in range(1, len(featureCols) + 1):
        selector = ChiSqSelector(numTopFeatures=i, featuresCol="features",
                                 outputCol="selectedFeatures", labelCol="label")

        selectedData = selector.fit(selectorData).transform(selectorData)
        trainSelected, testSelected = selectedData.randomSplit([0.7, 0.3])
        rfModel = rf.fit(trainSelected)

        prediction = rfModel.transform(testSelected)
        evaluator = BinaryClassificationEvaluator()
        measure = np.vstack([evaluateLr(prediction, evaluator, i), measure])
    return measure

def getFeatureCols(num_cols, cat_cols):
    featureCols = []
    for i in num_cols:
        featureCols.append(i + "scaled")

    for i in cat_cols:
        featureCols.append(i + "_hoted")
    return featureCols;


def ProcessData(pandaData, pipeline):
    sparkData = sqlContext.createDataFrame(pandaData)
    transformedData = pipeline.fit(sparkData).transform(sparkData)
    return transformedData


def getNumericCols(pandaData):
    target_col = ["Churn"]
    cat_cols = pandaData.nunique()[pandaData.nunique() < 6].keys().tolist()
    cat_cols = [x for x in cat_cols if x not in target_col]
    num_cols = [x for x in pandaData.columns if x not in cat_cols + target_col]
    return num_cols

def getCategoricCols(pandaData):
    target_col = ["Churn"]
    cat_cols = pandaData.nunique()[pandaData.nunique() < 6].keys().tolist()
    cat_cols = [x for x in cat_cols if x not in target_col]
    return cat_cols

if __name__ == "__main__":

    spark = SparkSession.builder.appName("Churn_Feature_Selection").getOrCreate()
    data = loadData(spark)
    pandaData = cleanData(data)

    num_cols = getNumericCols(pandaData)
    cat_cols = getCategoricCols(pandaData)

    pipeline = initializePipeline(num_cols, cat_cols)
    selectorData = ProcessData(pandaData,pipeline)

    lr = LogisticRegression(featuresCol='selectedFeatures', labelCol='label', maxIter=10)

    featureCols = getFeatureCols(num_cols, cat_cols)
    mesureTable =  getAllMeasure(lr, selectorData, featureCols)

    print("Measure obtenue avec la regression logistique \n", mesureTable)
    print(["number", " areaUnderRoc ", " areaUnderPR "])

    spark.stop()
