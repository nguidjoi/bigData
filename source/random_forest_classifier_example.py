#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Random Forest Classifier Example.
"""
from __future__ import print_function

import warnings

import matplotlib.pyplot as plt  # visualization
import pandas as pd
# $example on$
from IPython.core.display import display

warnings.filterwarnings("ignore")
import plotly.offline as py  # visualization

py.init_notebook_mode(connected=True)  # visualization
import plotly.graph_objs as go  # visualization
# $example off$
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import UserDefinedFunction

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("RandomForestClassifierExample") \
        .getOrCreate()

    # $example on$
    # Load and parse the data file, converting it to a DataFrame.
    data = spark.read.format("csv").option("header", "true").load("../data/dataset.csv").drop("customerID")
    display(data)
    # data.show()

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    # labelIndexer = StringIndexer(inputCol="Churn", outputCol="indexedChurn").fit(data)

    # labelIndexer = StringIndexer(inputCol="Churn", outputCol="idChurn").fit(data).transform(data).drop("Churn");
    # labelIndexer1 = StringIndexer(inputCol="PaymentMethod", outputCol="idPaymentMethod").fit(labelIndexer).transform(labelIndexer).drop("PaymentMethod");
    # labelIndexer2 = StringIndexer(inputCol="Churn", outputCol="idChurn").fit(data).transform(labelIndexer1).drop("Churn");

    numberMap = {'Month-to-month': 1.0, 'One year': 2.0, 'Two year': 3.0, 'Yes': 1.0, 'No': 0.0, 'Male': 1.0,
                 'Female': 0.0, 'No internet service': 0.0, 'No phone service': 0.0, 'Bank transfer (automatic)': 1.0,
                 'Credit card (automatic)': 2.0, 'Electronic check': 3.0, 'Mailed check': 4.0, 'Fiber optic': 1.0,
                 'DSL': 2.0}

    number = UserDefinedFunction(lambda k: numberMap[k], DoubleType())

    # data.withColumn("SeniorCitizen")tenure MonthlyCharges|TotalCharges
    data = data.withColumn("SeniorCitizen", data["SeniorCitizen"].cast(DoubleType())) \
        .withColumn("tenure", data["tenure"].cast(DoubleType())) \
        .withColumn("MonthlyCharges", data["MonthlyCharges"].cast(DoubleType())) \
        .withColumn("TotalCharges", data["TotalCharges"].cast(DoubleType())) \
        .withColumn("Partner", number('Partner')) \
        .withColumn("Dependents", number('Dependents')) \
        .withColumn("PaperlessBilling", number('PaperlessBilling')) \
        .withColumn("Churn", number('Churn')) \
        .withColumn("PhoneService", number('PhoneService')) \
        .withColumn("gender", number('gender')) \
        .withColumn("InternetService", number('InternetService')) \
        .withColumn("MultipleLines", number('MultipleLines')) \
        .withColumn("OnlineSecurity", number('OnlineSecurity')) \
        .withColumn("OnlineBackup", number('OnlineBackup')) \
        .withColumn("DeviceProtection", number('DeviceProtection')) \
        .withColumn("TechSupport", number('TechSupport')) \
        .withColumn("StreamingTV", number('StreamingTV')) \
        .withColumn("Contract", number('Contract')) \
        .withColumn("PaymentMethod", number('PaymentMethod')) \
        .withColumn("StreamingMovies", number('StreamingMovies'))

    data.show()
    datas = data.drop('Churn')
    data1 = datas.sample(False, 0.10).toPandas()
    data1.to_html()
    dps = pd.DataFrame(data1, columns=['MonthlyCharges', 'TotalCharges', 'tenure', 'SeniorCitizen'])
    axs = pd.scatter_matrix(dps, alpha=0.2);
    dps.corr(method='spearman')

    plt.show()

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

    # r2 = Correlation.corr(datas, "features", "spearman").head()
    # print("Spearman correlation matrix:\n" + str(r2[0]))
    # string_features = [t[0] for t in data.dtypes if t[1] == 'string' or t[1] == 'String']
    # display( string_features)

    # strData.show()
    # indexers = [StringIndexer(inputCol=column, outputCol=column +"id") for column in
    #            list(set(string_features) )]

    # pipeline = Pipeline(stages=indexers)
    # df_r = pipeline.fit(data)

    # df_r.transform(data)

    # df_r.show()

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    # featureIndexer =\
    #    VectorIndexer(inputCol="gender", outputCol="genderNum", maxCategories=4).fit(data)

    # featureIndexer.show()
    # Split the data into training and test sets (30% held out for testing)
    # (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    # rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

    # Convert indexed labels back to original labels.
    # labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
    #                               labels=labelIndexer.labels)

    # Chain indexers and forest in a Pipeline
    # pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

    # Train model.  This also runs the indexers.
    # model = pipeline.fit(trainingData)

    # Make predictions.
    # predictions = model.transform(testData)

    # Select example rows to display.
    # predictions.select("predictedLabel", "label", "features").show(5)

    # Select (prediction, true label) and compute test error
    # evaluator = MulticlassClassificationEvaluator(
    #    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    # accuracy = evaluator.evaluate(predictions)
    # print("Test Error = %g" % (1.0 - accuracy))

    # rfModel = model.stages[2]
    # print(rfModel)  # summary only
    # $example off$

    spark.stop()
