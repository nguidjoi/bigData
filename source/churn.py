# Importing libraries
import warnings

# Input data files are available in the "../input/" directory.
import matplotlib.pyplot as plt  # visualization
import numpy as np  # linear algebra
import source.dataPreprocessing as process

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pandas as pd
from pandas.plotting import scatter_matrix
import source.FeatureSelection as featureSelector
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.sql import SparkSession
import seaborn as sn
warnings.filterwarnings("ignore")
import plotly.offline as py
py.init_notebook_mode(connected=True)

def crossvalidate(lr, evaluator, train, test):
    paramGrid = (ParamGridBuilder()
                 .addGrid(lr.regParam, [0.01, 0.5, 2.0])
                 .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                 .addGrid(lr.maxIter, [1, 5, 10])
                 .build())

    cvLr = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
    cvLrModel = cvLr.fit(train)
    predictions = cvLrModel.transform(test)
    print('Cross validation Test Area Under ROC', evaluator.evaluate(predictions))


if __name__ == "__main__":
    spark = SparkSession.builder.appName("Churn_data_mining").getOrCreate()

    sparkData = featureSelector.loadData(spark)
    pandasData = featureSelector.cleanData(sparkData)
    num_cols = featureSelector.getNumericCols(pandasData)
    fdata = process.preprocesData(sparkData)

    trainingData, testData = fdata.randomSplit([0.3, 0.3])

    # Train a  model.
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)
    #lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10)

    #lrModel = lr.fit(trainingData)
    rfModel = rf.fit(trainingData)

    rfTransformed = rfModel.transform(testData)
    #rlTransformed = lrModel.transform(testData)

    # evaluate
    evaluator = BinaryClassificationEvaluator()

    #print('Evaluation for Logistic regression \n')
    #print(featureSelector.evaluatePrediction(rlTransformed, evaluator))
    print('Evaluation for random forest \n' )
    print(featureSelector.evaluatePrediction( rfTransformed, evaluator))

spark.stop()
