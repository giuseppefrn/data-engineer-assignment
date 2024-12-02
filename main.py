import argparse
import logging
import re
from datetime import datetime
from typing import Union

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

# Set up logger
log_level = logging.DEBUG
logger = logging.getLogger("DataEngineerAssignment")
logger.setLevel(log_level)

# Stream handler for console output
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)

# Formatter for log messages
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add handler to the logger
logger.addHandler(console_handler)

spark = SparkSession.builder.appName("DataEngineerAssignment").getOrCreate()


def to_snake_case(name: str) -> str:
    """
    Convert a string to snake_case
    :param name: input string
    :return: snake_case string
    """
    snake_case_name = re.sub(r"[-\s]", "_", name).lower()
    logger.debug(f"Converted {name} to '{snake_case_name}'.")
    return snake_case_name


def add_metadata_columns(df: DataFrame, file_path: str) -> DataFrame:
    """
    Add metadata columns to a DataFrame
    :param df:
    :param file_path:
    :return:
    """
    logger.info("Adding metadata columns: file_path, execution_datetime...")
    execution_datetime = datetime.now().isoformat()
    return df.withColumn("file_path", F.lit(file_path)).withColumn(
        "execution_datetime", F.lit(execution_datetime)
    )


def write_df(
    output_path: str, df: DataFrame, method: str, partition_by: Union[list[str], None]
) -> None:
    """
    Write a DataFrame to a given output path partitioned by year, month, and day
    :param output_path:
    :param df:
    :param method:
    :param partition_by:
    :return:
    """
    logger.info(
        f"Writing DataFrame to {output_path}, method: {method}, partition_by: {partition_by}..."
    )
    if partition_by is not None:
        df.write.mode(method).partitionBy(partition_by).parquet(output_path)
    else:
        df.write.mode(method).parquet(output_path)


def process_bronze_data(df: DataFrame, output_path: str) -> DataFrame:
    """
    Process the raw data to create the bronze table
    with snake_case column names and stores it
    :param df: Raw DataFrame
    :param output_path: Path to save the dataset
    :return:
    """
    bronze_df = add_metadata_columns(df, output_path)

    logger.info("Renaming columns to snake_case...")
    snake_case_columns = [to_snake_case(column) for column in bronze_df.columns]
    bronze_df = bronze_df.toDF(*snake_case_columns)
    logger.info("Column renaming completed.")

    # Partition the data
    if "order_date" not in bronze_df.columns:
        raise ValueError("The 'order_date' column is missing in the dataset.")

    logger.info("Parsing 'order_date' into a proper date format...")
    bronze_df = bronze_df.withColumn(
        "order_date", F.to_date(F.col("order_date"), "dd/MM/yyyy")
    )

    logger.info("Extracting year, month, and day from 'order_date'...")
    bronze_df = (
        bronze_df.withColumn("year", F.year(F.col("order_date")))
        .withColumn("month", F.month(F.col("order_date")))
        .withColumn("day", F.dayofmonth(F.col("order_date")))
    )

    write_df(output_path, bronze_df, "ignore", ["year", "month", "day"])
    return bronze_df


def process_silver_data(bronze_df: DataFrame, output_path: str) -> DataFrame:
    """
    Process the Bronze dataset into a refined Silver dataset.
    :param bronze_df: Bronze DataFrame
    :param output_path: Path to save the dataset
    :return: Silver DataFrame
    """
    logger.info("Starting Silver layer processing...")

    silver_df = add_metadata_columns(bronze_df, output_path)

    # Convert `ship_date` to date format
    silver_df = silver_df.withColumn("ship_date", F.to_date("ship_date", "dd/MM/yyyy"))

    # Drop rows with NULLs in critical columns
    silver_df = silver_df.dropna(
        subset=["order_date", "customer_id", "customer_name", "order_id"]
    )

    # Split `customer_name` into `customer_first_name` and `customer_last_name`
    logger.info(
        "Splitting 'customer_name' into 'customer_first_name' and 'customer_last_name'..."
    )
    silver_df = (
        silver_df.withColumn(
            "customer_first_name", F.split(F.col("customer_name"), " ").getItem(0)
        )
        .withColumn(
            "customer_last_name",
            F.expr(
                "slice(split(customer_name, ' '), 2, size(split(customer_name, ' ')))"
            ),
        )
        .withColumn("customer_last_name", F.expr("array_join(customer_last_name, ' ')"))
    )

    # Save the Silver DataFrame
    write_df(output_path, silver_df, "ignore", ["year", "month", "day"])
    return silver_df


def process_sales_data(
    silver_df: DataFrame, columns: list[str], output_path: str
) -> None:
    """
    Extract, prepare, and write the Sales dataset for the Gold layer.
    :param silver_df: Silver DataFrame
    :param columns: Columns to select for the Sales dataset
    :param output_path: Path to save the Gold Sales dataset
    :return: None
    """
    logger.info("Extracting Sales dataset from Silver layer...")

    sales_df = add_metadata_columns(silver_df, output_path)

    # Select relevant columns for the Sales dataset
    sales_df = sales_df.select(
        columns + ["year", "month", "day", "execution_datetime", "file_path"]
    )

    # Write the dataset to the Gold layer partitioned by year, month, and day
    write_df(output_path, sales_df, "overwrite", ["year", "month", "day"])


def process_gold_customer(
    silver_df: DataFrame, columns: list[str], output_path: str
) -> None:
    """
    Extract and prepare the Customer dataset for the Gold layer.
    :param silver_df: Silver DataFrame
    :param columns: Columns to group by
    :param output_path: Path to save the dataset
    :return: None
    """
    logger.info("Extracting Customer dataset from Silver layer...")

    customer_df = add_metadata_columns(silver_df, output_path)

    latest_date_row = silver_df.agg(F.max("order_date").alias("latest_date")).collect()[
        0
    ]
    latest_date = latest_date_row["latest_date"]
    reference_date = F.lit(latest_date)

    logger.debug(f"Last date in the dataset: {latest_date}")

    # Calculate quantities for different time ranges
    customer_df = customer_df.groupBy(
        columns + ["execution_datetime", "file_path"]
    ).agg(
        F.countDistinct("order_id").alias("total_quantity_of_orders"),
        F.sum(
            F.when(F.datediff(reference_date, F.col("order_date")) <= 30, 1).otherwise(
                0
            )
        ).alias("quantity_last_month"),
        F.sum(
            F.when(F.datediff(reference_date, F.col("order_date")) <= 180, 1).otherwise(
                0
            )
        ).alias("quantity_last_6_months"),
        F.sum(
            F.when(F.datediff(reference_date, F.col("order_date")) <= 365, 1).otherwise(
                0
            )
        ).alias("quantity_last_12_months"),
    )

    write_df(output_path, customer_df, "overwrite", None)


def main(args: argparse.Namespace) -> None:
    """
    Main function
    :param args: argparse.Namespace
    :return: None
    """
    sales_columns = ["order_id", "order_date", "ship_date", "ship_mode", "city"]
    customer_columns = [
        "customer_id",
        "customer_first_name",
        "customer_last_name",
        "segment",
        "country",
    ]

    try:
        logger.info(f"Reading data from {args.input_path}...")
        df = spark.read.csv(args.input_path, header=True, inferSchema=True)

        logger.info("Dataset loaded successfully.")

        logger.info("Processing bronze data...")
        bronze_df = process_bronze_data(df, "outputs/bronze")

        logger.info("Processing silver data...")
        silver_df = process_silver_data(bronze_df, "outputs/silver")

        logger.info("Processing sales data...")
        process_sales_data(silver_df, sales_columns, "outputs/gold/sales")

        logger.info("Processing customer data...")
        process_gold_customer(silver_df, customer_columns, "outputs/gold/customer")

        logger.info("Process completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    # Parse arguments such as input file path
    parser = argparse.ArgumentParser(description="Data Engineer Assignment")
    parser.add_argument(
        "--input_path", default="data/train.csv", type=str, help="Input file path"
    )
    args = parser.parse_args()

    main(args)
