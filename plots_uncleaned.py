"""
NYC Yellow Taxi 2025 — Pre-Cleaning EDA Plots
Runs on the RAW parquet BEFORE any cleaning steps.

Fixes vs original:
  1. Duration uses F.unix_timestamp() — timestamp_ntz cannot cast to long directly
  2. RatecodeID comparisons cast to LongType (column is LongType in this dataset)
  3. Airport_fee referenced with capital A
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

# ── Config ────────────────────────────────────────────────────────────────────
RAW_PATH = "./YellowTripData/"
PLOT_DIR = "./plots_raw/"
SAMPLE_N = 200_000
SEED     = 42

os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
ACCENT  = "#4C72B0"
ACCENT2 = "#DD8452"
RED     = "#C44E52"
GRAY    = "#888888"

def save(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

# ── Spark ─────────────────────────────────────────────────────────────────────
spark = SparkSession.builder.appName("TaxiRawEDA").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(RAW_PATH)
total_rows = df.count()
print(f"\nRaw row count: {total_rows:,}")

# ── Derive duration & flags on raw data (no filtering) ────────────────────────
NEG_COLS = ["tolls_amount", "congestion_surcharge", "cbd_congestion_fee"]

df = (
    df
    # FIX: use unix_timestamp() — timestamp_ntz cannot cast to long directly
    .withColumn(
        "trip_duration_seconds",
        (F.unix_timestamp("tpep_dropoff_datetime") -
         F.unix_timestamp("tpep_pickup_datetime")).cast(IntegerType())
    )
    .withColumn("trip_duration_min", F.col("trip_duration_seconds") / 60.0)
    # Any of the three fee columns negative
    .withColumn(
        "has_negative_fee",
        F.when(
            F.greatest(*[
                F.when(F.col(c) < 0, F.lit(1)).otherwise(F.lit(0)) for c in NEG_COLS
            ]) > 0,
            F.lit(1)
        ).otherwise(F.lit(0))
    )
    # No-meter: distance == 0 AND fare == 0
    .withColumn(
        "no_meter",
        F.when(
            (F.col("trip_distance") == 0) & (F.col("fare_amount") == 0),
            F.lit(1)
        ).otherwise(F.lit(0))
    )
    # Speed proxy — raw, uncapped
    .withColumn(
        "speed_mph_proxy",
        F.when(
            F.col("trip_duration_seconds") > 0,
            F.round(
                F.col("trip_distance") / (F.col("trip_duration_seconds") / 3600.0), 2
            )
        ).otherwise(F.lit(None).cast("double"))
    )
    .withColumn("pickup_hour",      F.hour("tpep_pickup_datetime"))
    .withColumn("pickup_dayofweek", F.dayofweek("tpep_pickup_datetime"))
    .withColumn("pickup_month",     F.month("tpep_pickup_datetime"))
)

# ── Exact aggregates from full Spark dataset (before sampling) ────────────────

flag_counts = (
    df.groupBy("has_negative_fee", "no_meter")
      .count()
      .toPandas()
)
flag_counts["label"] = flag_counts.apply(
    lambda r: f"neg_fee={int(r.has_negative_fee)}, no_meter={int(r.no_meter)}", axis=1
)
flag_counts = flag_counts.sort_values("count", ascending=True)

neg_col_counts = {c: df.filter(F.col(c) < 0).count() for c in NEG_COLS}
print("Negative counts per fee column:", neg_col_counts)

hour_agg = (
    df.groupBy("pickup_hour")
      .agg(F.mean("trip_duration_min").alias("mean_duration"), F.count("*").alias("n"))
      .toPandas().sort_values("pickup_hour")
)
dow_agg = (
    df.groupBy("pickup_dayofweek")
      .agg(F.mean("trip_duration_min").alias("mean_duration"), F.count("*").alias("n"))
      .toPandas().sort_values("pickup_dayofweek")
)
month_agg = (
    df.groupBy("pickup_month")
      .agg(F.mean("trip_duration_min").alias("mean_duration"), F.count("*").alias("n"))
      .toPandas().sort_values("pickup_month")
)

# ── Sample to Pandas ──────────────────────────────────────────────────────────
frac = min(1.0, SAMPLE_N / total_rows)
pdf  = df.sample(fraction=frac, seed=SEED).toPandas()
print(f"Sampled {len(pdf):,} rows for distribution plots.\n")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Raw target distribution
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 1: Raw target distribution …")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("Raw trip duration — before cleaning", fontsize=14, fontweight="bold", y=1.01)

dur = pdf["trip_duration_min"].dropna()

# Panel A — full range
ax = axes[0]
ax.hist(dur, bins=200, color=ACCENT, edgecolor="none")
ax.set_xlabel("Duration (minutes)")
ax.set_ylabel("Count")
ax.set_title("Full range")
ax.axvline(0, color=RED, linewidth=1.5, linestyle="--", label="0 min")
pct_neg = (dur < 0).mean() * 100
ax.text(0.55, 0.88, f"{pct_neg:.2f}% negative", transform=ax.transAxes,
        color=RED, fontsize=9)
ax.legend(frameon=False, fontsize=9)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# Panel B — zoomed to [-30, 200] min
ax2 = axes[1]
zoom = dur[(dur >= -30) & (dur <= 200)]
ax2.hist(zoom, bins=150, color=ACCENT2, edgecolor="none")
ax2.set_xlabel("Duration (minutes)")
ax2.set_ylabel("Count")
ax2.set_title("Zoomed: −30 to 200 min")
ax2.axvline(0,   color=RED,      linewidth=1.5, linestyle="--", label="0 min")
ax2.axvline(1,   color="green",  linewidth=1.5, linestyle=":",  label="1 min (lower bound)")
ax2.axvline(180, color="orange", linewidth=1.5, linestyle=":",  label="180 min (upper bound)")
ax2.legend(frameon=False, fontsize=8)

# Panel C — log scale tail
ax3 = axes[2]
pos_dur = dur[dur > 0]
ax3.hist(pos_dur, bins=200, color=GRAY, edgecolor="none")
ax3.set_yscale("log")
ax3.set_xlabel("Duration (minutes, positive only)")
ax3.set_ylabel("Count (log)")
ax3.set_title("Log scale — tail extent")
ax3.axvline(180, color="orange", linewidth=1.5, linestyle=":", label="180 min cutoff")
pct_above_3h = (pos_dur > 180).mean() * 100
ax3.text(0.45, 0.88, f"{pct_above_3h:.2f}% > 3 hours",
         transform=ax3.transAxes, color="orange", fontsize=9)
ax3.legend(frameon=False, fontsize=9)

fig.tight_layout()
save(fig, "01_raw_target_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Raw distance vs duration (hexbin)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 2: Raw distance vs duration …")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Raw trip distance vs duration", fontsize=14, fontweight="bold", y=1.01)

dist_clean = pdf[["trip_distance", "trip_duration_min"]].dropna()

ax = axes[0]
hb = ax.hexbin(dist_clean["trip_distance"], dist_clean["trip_duration_min"],
               gridsize=60, cmap="YlOrRd", mincnt=1, bins="log")
fig.colorbar(hb, ax=ax, label="log10(count)")
ax.set_xlabel("Trip distance (miles)")
ax.set_ylabel("Duration (minutes)")
ax.set_title("All raw data")
ax.axvline(80, color=RED,    linewidth=1.5, linestyle="--", label="Distance cap (80 mi)")
ax.axhline(0,  color="orange", linewidth=1.2, linestyle=":",  label="Duration = 0")
ax.legend(frameon=False, fontsize=9)

ax2 = axes[1]
mask = (
    (dist_clean["trip_distance"] >= 0) & (dist_clean["trip_distance"] <= 40) &
    (dist_clean["trip_duration_min"] >= 0) & (dist_clean["trip_duration_min"] <= 120)
)
hb2 = ax2.hexbin(dist_clean.loc[mask, "trip_distance"],
                 dist_clean.loc[mask, "trip_duration_min"],
                 gridsize=60, cmap="YlOrRd", mincnt=1, bins="log")
fig.colorbar(hb2, ax=ax2, label="log10(count)")
ax2.set_xlabel("Trip distance (miles)")
ax2.set_ylabel("Duration (minutes)")
ax2.set_title("Zoomed: 0–40 mi, 0–120 min")

fig.tight_layout()
save(fig, "02_raw_distance_vs_duration.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Negative fee flag analysis
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 3: Negative fee flag analysis …")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Negative fee analysis — raw data", fontsize=14, fontweight="bold", y=1.01)

# 3a — flag combination counts (exact from Spark)
ax = axes[0]
colors_bar = [ACCENT if (r.has_negative_fee == 0) else RED
              for _, r in flag_counts.iterrows()]
bars = ax.barh(flag_counts["label"], flag_counts["count"],
               color=colors_bar, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Row count")
ax.set_title("Flag combination counts")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
for bar, val in zip(bars, flag_counts["count"]):
    ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=8)

# 3b — per-column negative counts
ax2 = axes[1]
col_labels = [c.replace("_", "\n") for c in neg_col_counts.keys()]
col_vals   = list(neg_col_counts.values())
ax2.bar(col_labels, col_vals, color=[ACCENT2, RED, GRAY], edgecolor="white", linewidth=0.4)
ax2.set_ylabel("Row count with negative value")
ax2.set_title("Negative counts per fee column")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
for i, v in enumerate(col_vals):
    pct = v / total_rows * 100
    ax2.text(i, v * 1.01, f"{pct:.2f}%", ha="center", fontsize=9)

# 3c — duration density: negative-fee vs normal rows
ax3 = axes[2]
neg_fee_dur  = pdf.loc[pdf["has_negative_fee"] == 1, "trip_duration_min"].dropna()
norm_fee_dur = pdf.loc[pdf["has_negative_fee"] == 0, "trip_duration_min"].dropna()
clip_fn = lambda s: s[(s >= -30) & (s <= 200)]
ax3.hist(clip_fn(norm_fee_dur), bins=100, color=ACCENT, alpha=0.6,
         density=True, label="Normal fee rows")
ax3.hist(clip_fn(neg_fee_dur),  bins=100, color=RED,    alpha=0.7,
         density=True, label="Negative fee rows")
ax3.axvline(0, color="black", linewidth=1, linestyle="--")
ax3.set_xlabel("Duration (minutes, clipped −30 to 200)")
ax3.set_ylabel("Density")
ax3.set_title("Duration: negative vs normal fee rows")
ax3.legend(frameon=False, fontsize=9)

fig.tight_layout()
save(fig, "03_raw_negative_flag_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Raw feature correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 4: Raw correlation heatmap …")

RAW_CORR_COLS = [
    "trip_duration_seconds", "trip_distance", "passenger_count",
    "fare_amount", "extra", "tip_amount", "tolls_amount",
    "total_amount", "congestion_surcharge", "cbd_congestion_fee",
    "pickup_hour", "pickup_dayofweek", "pickup_month",
    "speed_mph_proxy", "has_negative_fee", "no_meter",
]
present = [c for c in RAW_CORR_COLS if c in pdf.columns]
# Coerce all to numeric — LongType columns may come through as object in older pandas
corr_pdf = pdf[present].apply(lambda col: col.astype(float, errors="ignore"))
corr = corr_pdf.corr()

fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, vmin=-1, vmax=1,
            linewidths=0.4, linecolor="white",
            annot_kws={"size": 7.5}, ax=ax)
ax.set_title("Pearson correlation — RAW features (pre-cleaning)",
             fontsize=14, fontweight="bold", pad=12)
fig.tight_layout()
save(fig, "04_raw_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Raw temporal patterns (exact Spark aggregates)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 5: Raw temporal patterns …")

DAY_LABELS   = {1:"Sun",2:"Mon",3:"Tue",4:"Wed",5:"Thu",6:"Fri",7:"Sat"}
MONTH_LABELS = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("Raw mean trip duration across time dimensions",
             fontsize=14, fontweight="bold", y=1.01)

# 5a — by hour
ax = axes[0]
ax.bar(hour_agg["pickup_hour"], hour_agg["mean_duration"],
       color=ACCENT, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Pickup hour (0–23)")
ax.set_ylabel("Mean duration (min) — raw")
ax.set_title("By hour of day")
ax.set_xticks(range(0, 24, 3))
for band in [(7, 9), (16, 19)]:
    ax.axvspan(band[0], band[1], alpha=0.12, color=RED,
               label="Rush hour" if band[0] == 7 else "")
ax.legend(frameon=False, fontsize=9)

# 5b — by day of week
dow_agg["day_label"] = dow_agg["pickup_dayofweek"].map(DAY_LABELS)
colors_dow = [RED if d in [1, 7] else ACCENT for d in dow_agg["pickup_dayofweek"]]
axes[1].bar(dow_agg["day_label"], dow_agg["mean_duration"],
            color=colors_dow, edgecolor="white", linewidth=0.4)
axes[1].set_xlabel("Day of week")
axes[1].set_ylabel("Mean duration (min) — raw")
axes[1].set_title("By day of week")
axes[1].legend(
    handles=[mpatches.Patch(facecolor=RED,    label="Weekend"),
             mpatches.Patch(facecolor=ACCENT, label="Weekday")],
    frameon=False, fontsize=9
)

# 5c — by month
month_agg["month_label"] = month_agg["pickup_month"].map(MONTH_LABELS)
axes[2].plot(month_agg["month_label"], month_agg["mean_duration"],
             marker="o", color=ACCENT2, linewidth=2, markersize=7)
axes[2].fill_between(month_agg["month_label"], month_agg["mean_duration"],
                     alpha=0.15, color=ACCENT2)
axes[2].set_xlabel("Month")
axes[2].set_ylabel("Mean duration (min) — raw")
axes[2].set_title("By month (2025)")
axes[2].tick_params(axis="x", rotation=45)

fig.tight_layout()
save(fig, "05_raw_temporal_patterns.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — Raw speed proxy distribution
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 6: Raw speed proxy distribution …")

speed = pdf["speed_mph_proxy"].dropna()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("Raw speed proxy distribution (mph) — before cleaning",
             fontsize=14, fontweight="bold", y=1.01)

# 6a — full range log scale
ax = axes[0]
ax.hist(speed, bins=300, color=ACCENT, edgecolor="none")
ax.set_yscale("log")
ax.set_xlabel("Speed (mph)")
ax.set_ylabel("Count (log)")
ax.set_title("Full range")
ax.axvline(60, color=RED, linestyle="--", linewidth=1.5, label="60 mph cap")
pct_above = (speed > 60).mean() * 100
ax.text(0.5, 0.88, f"{pct_above:.2f}% > 60 mph",
        transform=ax.transAxes, color=RED, fontsize=9)
ax.legend(frameon=False, fontsize=9)

# 6b — zoomed to 0–120 mph
ax2 = axes[1]
bulk = speed[(speed >= 0) & (speed <= 120)]
sns.histplot(bulk, bins=120, kde=True, color=ACCENT2, ax=ax2,
             line_kws={"linewidth": 2})
ax2.axvline(60, color=RED, linestyle="--", linewidth=1.5, label="60 mph cap")
p50 = bulk.median()
ax2.axvline(p50, color="green", linestyle=":", linewidth=1.5,
            label=f"Median {p50:.1f} mph")
ax2.set_xlabel("Speed (mph)")
ax2.set_ylabel("Count")
ax2.set_title("Zoomed: 0–120 mph")
ax2.legend(frameon=False, fontsize=9)

# 6c — speed by RatecodeID (LongType — compare as int, map works fine in pandas)
ax3 = axes[2]
# FIX: RatecodeID comes through as int64 in pandas from LongType — .isin() works directly
ratecode_map = {1: "Standard", 2: "JFK", 3: "Newark",
                4: "Nassau/West.", 5: "Negotiated", 6: "Group"}
plot_df = pdf[pdf["RatecodeID"].isin(ratecode_map.keys())].copy()
plot_df["rate_label"] = plot_df["RatecodeID"].map(ratecode_map)
plot_df = plot_df[plot_df["speed_mph_proxy"].between(0, 120)]
order = [v for k, v in sorted(ratecode_map.items())
         if v in plot_df["rate_label"].unique()]
sns.boxplot(data=plot_df, x="rate_label", y="speed_mph_proxy",
            order=order, palette="muted", ax=ax3,
            flierprops=dict(marker=".", markersize=2, alpha=0.3))
ax3.set_xlabel("Rate code")
ax3.set_ylabel("Speed proxy (mph, 0–120)")
ax3.set_title("Speed by rate code")
ax3.tick_params(axis="x", rotation=30)
ax3.axhline(60, color=RED, linestyle="--", linewidth=1.2, alpha=0.7)

fig.tight_layout()
save(fig, "06_raw_speed_proxy_distribution.png")

print(f"\nAll raw EDA plots saved to: {PLOT_DIR}")
spark.stop()