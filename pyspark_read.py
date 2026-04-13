"""
NYC Yellow Taxi 2025 — Data Cleaning Pipeline
Goal: predict trip_duration_seconds

Schema notes (actual parquet types):
  - passenger_count, RatecodeID, payment_type : LongType
  - tpep_pickup/dropoff_datetime               : timestamp_ntz (no timezone)
  - Airport_fee                                : capital A
  - All amount columns                         : DoubleType

Fixes vs original:
  1. Duration uses F.unix_timestamp() — timestamp_ntz cannot cast to long directly
  2. Airport_fee drop corrected (capital A)
  3. RatecodeID == 99 comparison cast to LongType to avoid mismatch
  4. total_amount < 0 rows dropped
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType

spark = SparkSession.builder.appName("NYCTaxiCleaning").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
df = spark.read.parquet("./YellowTripData/")

print(f"Raw row count: {df.count():,}")
df.printSchema()


# ─────────────────────────────────────────────
# 2. NEGATIVE VALUE ANALYSIS & REMOVAL
# ─────────────────────────────────────────────

neg_cols = ["tolls_amount", "congestion_surcharge", "cbd_congestion_fee"]

# Any of the three fee columns is negative
#in case of refund
has_negative = F.greatest(*[
    F.when(F.col(c) < 0, F.lit(1)).otherwise(F.lit(0)) for c in neg_cols
]) > 0

# No meter engaged: zero distance AND zero fare 

# also check stop and fwd column to avoid flagging cases where no meter was engaged but the trip was legitimately free (e.g. store_and_fwd_flag = Y for a canceled trip)
no_meter = (F.col("trip_distance") == 0) & (F.col("fare_amount") == 0)

# Overcharge suspicion: negative fee + total_amount above 75th percentile
q75 = df.approxQuantile("total_amount", [0.75], 0.01)[0]
overcharge_flag = has_negative & (F.col("total_amount") > q75)

df = (
    df
    .withColumn("flag_no_meter",   F.when(no_meter,        F.lit(1)).otherwise(F.lit(0)))
    .withColumn("flag_overcharge", F.when(overcharge_flag, F.lit(1)).otherwise(F.lit(0)))
    # Drop negatives only on no-meter / ghost trips
    .filter(~(has_negative & no_meter))
    # Drop rows where total_amount itself is negative (unrecoverable billing error)
    .filter(F.col("total_amount") >= 0)
)

print("\n=== After negative-value filtering ===")
print(f"Rows remaining: {df.count():,}")
print("Overcharge-flagged rows:",
      df.filter(F.col("flag_overcharge") == 1).count())


# ─────────────────────────────────────────────
# 3. TRIP DISTANCE OUTLIER REMOVAL
# ─────────────────────────────────────────────

q1, q3 = df.approxQuantile("trip_distance", [0.25, 0.75], 0.01)
iqr         = q3 - q1
upper_iqr   = q3 + 3.0 * iqr
domain_cap  = 250 # Based on domain knowledge: NYC trips >250 miles are almost certainly data errors (e.g. odometer reset, wrong units)
upper_bound = max(upper_iqr, domain_cap)

print(f"\n=== Trip distance bounds ===")
print(f"  Q1={q1:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}")
print(f"  IQR fence={upper_iqr:.2f}  domain cap={domain_cap}  applied={upper_bound:.2f}")

df = df.filter(
    (F.col("trip_distance") > 0) &
    (F.col("trip_distance") <= upper_bound)
)
print(f"Rows after distance filtering: {df.count():,}")


# ─────────────────────────────────────────────
# 4. DURATION DERIVATION & VALIDATION
#
# FIX: timestamp_ntz cannot be cast directly to long.
#      F.unix_timestamp() correctly handles timestamp_ntz columns.
# ─────────────────────────────────────────────

df = df.withColumn(
    "trip_duration_seconds",
    (F.unix_timestamp("tpep_dropoff_datetime") -
     F.unix_timestamp("tpep_pickup_datetime")).cast(IntegerType())
)

DURATION_MIN =    60    # 1 minute
DURATION_MAX = 10_800   # 3 hours

df = df.filter(
    (F.col("trip_duration_seconds") >= DURATION_MIN) &
    (F.col("trip_duration_seconds") <= DURATION_MAX)
)
print(f"\nRows after duration filtering: {df.count():,}")


# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────

# 5a. Temporal features
df = (
    df
    .withColumn("pickup_hour",      F.hour("tpep_pickup_datetime"))
    .withColumn("pickup_dayofweek", F.dayofweek("tpep_pickup_datetime"))  # 1=Sun…7=Sat
    .withColumn("pickup_month",     F.month("tpep_pickup_datetime"))
    .withColumn("is_weekend",
        F.when(F.dayofweek("tpep_pickup_datetime").isin([1, 7]), F.lit(1))
         .otherwise(F.lit(0)))
    .withColumn("is_rush_hour",
        F.when(
            (F.col("is_weekend") == 0) & (
                F.col("pickup_hour").between(7, 9) |
                F.col("pickup_hour").between(16, 19)),
            F.lit(1)
        ).otherwise(F.lit(0)))
)

# 5b. Categorical encoding
# store_and_fwd_flag: Y→1, N→0
df = df.withColumn(
    "store_fwd_flag_int",
    F.when(F.col("store_and_fwd_flag") == "Y", F.lit(1)).otherwise(F.lit(0))
)

# RatecodeID is LongType — compare as Long, then cast result to Int
df = df.withColumn(
    "ratecode_clean",
    F.when(
        F.col("RatecodeID") == F.lit(99).cast(LongType()),
        F.lit(None).cast(IntegerType())
    ).otherwise(F.col("RatecodeID").cast(IntegerType()))
)

# 5c. Speed proxy (mph)
df = df.withColumn(
    "speed_mph_proxy",
    F.round(
        F.col("trip_distance") / (F.col("trip_duration_seconds") / 3600.0),
        2
    )
)

# 5d. Drop leakage columns and raw columns replaced by engineered versions
# NOTE: the column name is Airport_fee (capital A) in this dataset
leakage_cols = [
    "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
    "improvement_surcharge", "total_amount", "congestion_surcharge",
    "Airport_fee", "cbd_congestion_fee",
    "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "store_and_fwd_flag",
    "RatecodeID",
]
df = df.drop(*leakage_cols)


# ─────────────────────────────────────────────
# 6. FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n=== Final cleaned dataset ===")
print(f"Rows : {df.count():,}")
print(f"Cols : {len(df.columns)}")
df.printSchema()
df.describe(
    "trip_duration_seconds", "trip_distance", "speed_mph_proxy",
    "passenger_count", "pickup_hour"
).show()

# ─────────────────────────────────────────────
# 7. SAVE
# ─────────────────────────────────────────────
OUTPUT_PATH = "/home/hetansh/Documents/532_FinalProject/YellowTripData_Cleaned/"
df.write.mode("overwrite").parquet(OUTPUT_PATH)
print(f"\nCleaned data written to: {OUTPUT_PATH}")