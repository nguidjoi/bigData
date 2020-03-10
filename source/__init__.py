
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
from pyspark.mllib.feature import ChiSqSelector
from pyspark.shell import sqlContext

from pyspark.sql import SparkSession
import seaborn as sn
from pyspark.sql.functions import UserDefinedFunction, udf

warnings.filterwarnings("ignore")
import plotly.offline as py  # visualization
py.init_notebook_mode(connected=True)  # visualization
from IPython.core.display import display
##from pyspark.ml.feature import ChiSqSelector
from pyspark.mllib.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors

if __name__ == "__main__":
    spark = SparkSession.builder.appName("RandomForestClassifierExample").getOrCreate()
    df = spark.createDataFrame([
        (7, Vectors.dense([0.0, 0.0, 18.0, 1.0]), 1.0,),
        (8, Vectors.dense([0.0, 1.0, 12.0, 0.0]), 0.0,),
        (9, Vectors.dense([1.0, 0.0, 15.0, 0.1]), 0.0,)], ["id", "features", "clicked"])

    selector = ChiSqSelector(numTopFeatures=2)

    result = selector.fit(df).transform(df)

    print("ChiSqSelector output with top %d features selected" % selector.getNumTopFeatures())
    result.show()