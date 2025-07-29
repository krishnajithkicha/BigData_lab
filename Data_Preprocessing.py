from pyspark.sql import SparkSession
from pyspark.sql.functions import col,isnan,when,count,countDistinct,min,max,mean,stddev
from pyspark.sql.functions import  trim, lower, avg, expr, log1p, sqrt
from pyspark.sql.functions import  lit, to_date
from pyspark.sql.types import NumericType
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StandardScaler, RobustScaler


spark=SparkSession.builder.appName("Data_preprocessing").getOrCreate()

data = [
    (1, " Alice ", 34, 70000),
    (2, "Bob", None, 60000),
    (3, "charlie", 29, None),
    (4, "David", 45, 80000),
    (5, "Eve", None, 55000)
]

df = spark.createDataFrame(data, schema=["id", "name", "age", "salary"])


def data_profilling():
    print("Schema")
    df.printSchema()
    print("Data sample")
    df.show()
    print("Total records:", df.count())
    print("Null count per column:")

    # Build the expression list safely
    exprs = []
    for field in df.schema.fields:
        c = field.name
        if isinstance(field.dataType, NumericType):
            exprs.append(count(when(col(c).isNull() | isnan(col(c)), c)).alias(c))
        else:
            exprs.append(count(when(col(c).isNull(), c)).alias(c))

    df.select(exprs).show()

    print("Distinct count per column:")
    df.agg(*[countDistinct(c).alias(c) for c in df.columns]).show()

    print("Basic statistics:")
    df.describe().show()

def data_cleaning(df):
    # 1. Handling Missing values
    # Calculate mean age and median salary for imputation
    mean_age = df.select(mean(col("age"))).first()[0]
    median_salary = df.approxQuantile("salary", [0.5], 0.01)[0]
    df = df.withColumn("age", when(col("age").isNull(), mean_age).otherwise(col("age")))
    df = df.withColumn("salary", when(col("salary").isNull(), median_salary).otherwise(col("salary")))
    print("Dataframe after using mean and median:")
    df.show(truncate=False)
    # 2. Remove duplicates
    df = df.dropDuplicates()
    print("DataFrame after removing duplicates:")
    df.show(truncate=False)
    # 3. Clean 'name' column: trim spaces and standardize case (capitalize first letter)
    df = df.withColumn("name", expr("initcap(trim(name))"))
    print("DataFrame after standardizing:")
    df.show(truncate=False)

    # 4. Outlier detection and removal on 'salary' using IQR
    quantiles = df.approxQuantile("salary", [0.25, 0.75], 0.01)
    Q1, Q3 = quantiles
    IQR = Q3 - Q1
    df = df.filter((col("salary") >= (Q1 - 1.5 * IQR)) & (col("salary") <= (Q3 + 1.5 * IQR)))
    print("Cleaned DataFrame after outlier detection:")
    df.show(truncate=False)
    return df

def data_reduction(df):

    assembler = VectorAssembler(inputCols=["age", "salary"], outputCol="features")
    df_vector = assembler.transform(df)

    # --- MinMaxScaler ---
    minmax_scaler = MinMaxScaler(inputCol="features", outputCol="minmax_scaled")
    minmax_model = minmax_scaler.fit(df_vector)
    df_minmax = minmax_model.transform(df_vector)

    # --- StandardScaler ---
    standard_scaler = StandardScaler(inputCol="features", outputCol="standard_scaled", withMean=True, withStd=True)
    standard_model = standard_scaler.fit(df_vector)
    df_standard = standard_model.transform(df_minmax)

    # --- RobustScaler ---
    robust_scaler = RobustScaler(inputCol="features", outputCol="robust_scaled")
    robust_model = robust_scaler.fit(df_vector)
    df_robust = robust_model.transform(df_standard)

    # Show results
    df_robust.select(
        "id", "name", "age", "salary",
        "minmax_scaled",
        "standard_scaled",
        "robust_scaled"
    ).show(truncate=False)

#DATA TRANSFORMATION
def apply_power_transform(df):
    df = df.withColumn("log_age", log1p(col("age")))
    df = df.withColumn("log_salary", log1p(col("salary")))
    print("Data after power transform (log1p):")
    df.show(truncate=False)
    return df

def apply_function_transformer(df):
    df = df.withColumn("sqrt_age", sqrt(col("age")))
    df = df.withColumn("reciprocal_salary", when(col("salary") != 0, 1 / col("salary")).otherwise(None))
    print("Data after function transforms:")
    df.show(truncate=False)
    return df


# Call the function
if __name__ == "__main__":
    data_profilling()
    cleaned_df = data_cleaning(df)
    power_df = apply_power_transform(cleaned_df)
    func_transformed_df = apply_function_transformer(power_df)
    print("Final data after all transformations and scaling:")
    data_reduction(cleaned_df)       # Pass cleaned df to data_reduction
    spark.stop()

