from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.appName("RandomForestClassifierExample").getOrCreate()
    # $example on$
    # Load and parse the data file, converting it to a DataFrame.
    df = spark.read.format("csv").option("header", "true").load("../data/dataset.csv").drop("customerID").drop_duplicates()

    stringIndexer = StringIndexer(inputCol="gender", outputCol="categoryIndex")
    model = stringIndexer.fit(df)
    indexed = model.transform(df)

    encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
    encoded = encoder.transform(indexed)
    encoded.show()