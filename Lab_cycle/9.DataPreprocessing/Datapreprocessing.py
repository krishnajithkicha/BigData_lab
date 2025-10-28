from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count, countDistinct, mean, expr, log1p, sqrt
from pyspark.sql.types import NumericType
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StandardScaler, RobustScaler
import sys


def create_spark_session():
    return SparkSession.builder.appName("Data_preprocessing").getOrCreate()


def data_profilling(df):
    print("\n=== Schema ===")
    df.printSchema()

    print("\n=== Data sample ===")
    df.show()

    print(f"Total records: {df.count()}")
    print("\n=== Null count per column ===")

    exprs = []
    for field in df.schema.fields:
        c = field.name
        if isinstance(field.dataType, NumericType):
            exprs.append(count(when(col(c).isNull() | isnan(col(c)), c)).alias(c))
        else:
            exprs.append(count(when(col(c).isNull(), c)).alias(c))
    df.select(exprs).show()

    print("\n=== Distinct count per column ===")
    df.agg(*[countDistinct(c).alias(c) for c in df.columns]).show()

    print("\n=== Basic statistics ===")
    df.describe().show()


def data_cleaning(df):
    # Handle Missing Values
    mean_age = df.select(mean(col("age"))).first()[0]
    median_salary = df.approxQuantile("salary", [0.5], 0.01)[0]

    df = df.withColumn("age", when(col("age").isNull(), mean_age).otherwise(col("age")))
    df = df.withColumn("salary", when(col("salary").isNull(), median_salary).otherwise(col("salary")))
    print("\n=== After mean/median imputation ===")
    df.show(truncate=False)

    # Remove Duplicates
    df = df.dropDuplicates()
    print("\n=== After removing duplicates ===")
    df.show(truncate=False)

    # Clean name column
    df = df.withColumn("name", expr("initcap(trim(name))"))
    print("\n=== After standardizing names ===")
    df.show(truncate=False)

    # Outlier removal (IQR)
    quantiles = df.approxQuantile("salary", [0.25, 0.75], 0.01)
    Q1, Q3 = quantiles
    IQR = Q3 - Q1
    df = df.filter((col("salary") >= (Q1 - 1.5 * IQR)) & (col("salary") <= (Q3 + 1.5 * IQR)))
    print("\n=== After removing outliers ===")
    df.show(truncate=False)
    return df


def data_reduction(df):
    assembler = VectorAssembler(inputCols=["age", "salary"], outputCol="features")
    df_vector = assembler.transform(df)

    # MinMaxScaler
    minmax_scaler = MinMaxScaler(inputCol="features", outputCol="minmax_scaled")
    minmax_model = minmax_scaler.fit(df_vector)
    df_minmax = minmax_model.transform(df_vector)

    # StandardScaler
    standard_scaler = StandardScaler(inputCol="features", outputCol="standard_scaled", withMean=True, withStd=True)
    standard_model = standard_scaler.fit(df_vector)
    df_standard = standard_model.transform(df_minmax)

    # RobustScaler
    robust_scaler = RobustScaler(inputCol="features", outputCol="robust_scaled")
    robust_model = robust_scaler.fit(df_vector)
    df_robust = robust_model.transform(df_standard)

    print("\n=== Scaled Data ===")
    df_robust.select(
        "id", "name", "age", "salary", "minmax_scaled", "standard_scaled", "robust_scaled"
    ).show(truncate=False)


def apply_power_transform(df):
    df = df.withColumn("log_age", log1p(col("age")))
    df = df.withColumn("log_salary", log1p(col("salary")))
    print("\n=== After Power Transform (log1p) ===")
    df.show(truncate=False)
    return df


def apply_function_transformer(df):
    df = df.withColumn("sqrt_age", sqrt(col("age")))
    df = df.withColumn("reciprocal_salary", when(col("salary") != 0, 1 / col("salary")).otherwise(None))
    print("\n=== After Function Transformations ===")
    df.show(truncate=False)
    return df


def main(file_path):
    spark = create_spark_session()
    print(f"Reading file: {file_path}")
    df = spark.read.csv(file_path, header=True, inferSchema=True)

    data_profilling(df)
    cleaned_df = data_cleaning(df)
    power_df = apply_power_transform(cleaned_df)
    func_transformed_df = apply_function_transformer(power_df)
    data_reduction(cleaned_df)

    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: spark-submit data_preprocessing.py <input_file.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    main(input_file)
