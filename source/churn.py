# Importing libraries
import warnings

# Input data files are available in the "../input/" directory.
import matplotlib.pyplot as plt  # visualization
import numpy as np  # linear algebra

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric
import pandas as pd
from pandas.plotting import scatter_matrix

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.feature import StringIndexer, MinMaxScaler, VectorAssembler
from pyspark.shell import sqlContext

from pyspark.sql import SparkSession
import seaborn as sn
from pyspark.sql.functions import UserDefinedFunction, udf

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

    # Replacing spaces with null values in total charges column
    telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ", 999999999999999999)

    # Dropping null values from total charges column which contain .15% missing data
    telcom = telcom[telcom["TotalCharges"] != 999999999999999999]
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

    pd.DataFrame(data.take(5), columns=telcom.columns).transpose()
    data.select(num_cols).describe().toPandas().transpose()

    dps = pd.DataFrame(telcom, columns=num_cols)
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

    cordfs = dps.corr()
    plt.show()
    # labels
    lab = telcom["Churn"].value_counts().keys().tolist()
    # values
    val = telcom["Churn"].value_counts().values.tolist()

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

    print("\nFeatures featureCols : \n", featureCols)
    print("\nFeatures cat_cols_index : \n", cat_cols_index)
    print("\nFeatures cat_cols : \n", cat_cols)
    print("\nFeatures num_cols : \n", num_cols)

    featureAssembler = [VectorAssembler(inputCols=featureCols, outputCol= "features") ]

    pipeline = Pipeline(stages=indexers+assembler+normalizers+featureAssembler+labelindexers)
    fdata = pipeline.fit(dfs).transform(dfs)
    fdata.cache()

    (trainingData, testData) = fdata.randomSplit([0.7, 0.3])

    # Train a  model.
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)
    lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10)

    lrModel = lr.fit(trainingData)
    rfModel = rf.fit(trainingData)

    rfTransformed = rfModel.transform(testData)
    rlTransformed = lrModel.transform(testData)

    rfTransformed.s
    # evaluate

    def displayAccuracy(lrModel):
        beta = np.sort(lrModel.coefficients)
        plt.plot(beta)
        plt.ylabel('Beta Coefficients')
        plt.show()

        trainingSummary = lrModel.summary
        roc = trainingSummary.roc.toPandas()
        plt.plot(roc['FPR'], roc['TPR'])
        plt.ylabel('False Positive Rate')
        plt.xlabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

        f = trainingSummary.fMeasureByThreshold.toPandas()
        plt.plot(f['threshold'], f['F-Measure'])
        plt.ylabel('F-Measure')
        plt.xlabel('Threshold')
        plt.show()

        f = trainingSummary.fMeasureByThreshold.toPandas()
        plt.plot(f['threshold'], f['F-Measure'])
        plt.ylabel('F-Measure')
        plt.xlabel('Threshold')
        plt.show()

    def evaluates(rfModel, rfTransformed, evaluator):
        accuracy = evaluator.evaluate(rfTransformed)
        #modelAccuracy = evaluator.evaluate(rfModel)
        #print("Model AUC ROC accuracy : ", modelAccuracy)
        print("Prediction AUC ROC accuracy : ", accuracy)

    def evaluateLr(rfTransformed, evaluator):
        accuracy = evaluator.evaluate(rfTransformed)
        trainingSummary = lrModel.summary

        print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))
        print("Prediction AUC ROC accuracy : ", accuracy)

    def crossvalidate(lr, evaluator,train, test):
        # Create ParamGrid for Cross Validation
        paramGrid = (ParamGridBuilder()
                     .addGrid(lr.regParam, [0.01, 0.5, 2.0])
                     .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                     .addGrid(lr.maxIter, [1, 5, 10])
                     .build())

        cvLr = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
        cvLrModel = cvLr.fit(train)
        predictions = cvLrModel.transform(test)
        print('Cross validation Test Area Under ROC', evaluator.evaluate(predictions))

    def displayConfusionMatrix(df_cm):
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
        plt.show()

    #displayConfusionMatrix(rlTransformed)
    #displayConfusionMatrix(rfTransformed)
    evaluator = BinaryClassificationEvaluator()

    print('Evaluation for Logistic regression')
    evaluateLr(rlTransformed, evaluator)
    print('Evaluation for random forest')
    evaluates(rfModel, rfTransformed, evaluator)
    #crossvalidate(rf, evaluator, trainingData, testData)

    print('Cross validation Evaluation for Logistic regression')
    crossvalidate(lr, evaluator, trainingData, testData)
    displayAccuracy(lrModel)

spark.stop()
