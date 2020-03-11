from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.appName("RandomForestClassifierExample").getOrCreate()
    # $example on$
    # Load and parse the data file, converting it to a DataFrame.
    df = spark.read.format("csv").option("header", "true").load("../data/dataset.csv").drop("customerID").drop_duplicates()
    telcom =df.toPandas()

    # Replacing spaces with null values in total charges column
    telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ", 999999)

    # Dropping null values from total charges column which contain .15% missing data
    telcom = telcom[telcom["TotalCharges"] != 999999]
    telcom = telcom.reset_index()[telcom.columns]

    target_col = ["Churn"]
    cat_cols = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
    cat_cols = [x for x in cat_cols if x not in target_col]
    num_cols = [x for x in telcom.columns if x not in cat_cols + target_col]
    bin_cols = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
    multi_cols = [i for i in cat_cols if i not in bin_cols]

    encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
    target_col = ["Churn"]

    stringIndexer = StringIndexer(inputCol="gender", outputCol="categoryIndex")
    model = stringIndexer.fit(df)
    indexed = model.transform(df)

    encoded = encoder.transform(indexed)
    encoded.show()