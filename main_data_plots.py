"""
NYC Yellow Taxi 2025 — Post-Cleaning EDA Plots
Reads the cleaned parquet output from taxi_cleaning.py.

Fixes vs original:
  1. Coerce Long-typed columns (passenger_count, ratecode_clean etc.)
     to float before correlation — avoids pandas dtype mismatch in .corr()
  2. trip_duration_min derived from trip_duration_seconds (already IntegerType)
     so no timestamp casting needed here
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

# ── Config ────────────────────────────────────────────────────────────────────
CLEANED_PATH = os.getenv("TAXI_CLEANED_PATH", "./Data/Data_Cleaned")
PLOT_DIR     = "./plots/"
SAMPLE_N     = 200_000
SEED         = 42

os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
ACCENT  = "#4C72B0"
ACCENT2 = "#DD8452"
RED     = "#C44E52"

def save(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

# ── Spark ─────────────────────────────────────────────────────────────────────
spark = SparkSession.builder.appName("TaxiPlots").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

df_spark = spark.read.parquet(CLEANED_PATH)

frac = min(1.0, SAMPLE_N / df_spark.count())
pdf  = df_spark.sample(fraction=frac, seed=SEED).toPandas()

# Derive minutes column (trip_duration_seconds is IntegerType — safe arithmetic)
pdf["trip_duration_min"] = pdf["trip_duration_seconds"] / 60.0

print(f"\nSampled {len(pdf):,} rows for plotting.\n")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Target distribution
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 1: Target distribution …")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("Trip duration distribution", fontsize=14, fontweight="bold", y=1.01)

ax = axes[0]
sns.histplot(pdf["trip_duration_min"], bins=80, kde=True,
             color=ACCENT, ax=ax, line_kws={"linewidth": 2})
ax.set_xlabel("Duration (minutes)")
ax.set_ylabel("Count")
ax.set_title("Linear scale")
p50 = pdf["trip_duration_min"].median()
ax.axvline(p50, color=RED, linestyle="--", linewidth=1.5, label=f"Median {p50:.1f} min")
ax.legend(frameon=False)

ax2 = axes[1]
sns.histplot(pdf["trip_duration_min"], bins=80, kde=True,
             color=ACCENT2, ax=ax2, line_kws={"linewidth": 2})
ax2.set_yscale("log")
ax2.set_xlabel("Duration (minutes)")
ax2.set_ylabel("Count (log)")
ax2.set_title("Log scale (tail detail)")

fig.tight_layout()
save(fig, "01_target_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Trip distance vs duration (hexbin)
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 2: Distance vs duration …")

plot2 = pdf[["trip_distance", "trip_duration_min"]].dropna()

fig, ax = plt.subplots(figsize=(8, 6))
hb = ax.hexbin(plot2["trip_distance"], plot2["trip_duration_min"],
               gridsize=60, cmap="YlOrRd", mincnt=1, bins="log")
fig.colorbar(hb, ax=ax, label="log10(count)")
ax.set_xlabel("Trip distance (miles)")
ax.set_ylabel("Duration (minutes)")
ax.set_title("Trip distance vs duration — hexbin density", fontweight="bold")

m, b = np.polyfit(plot2["trip_distance"], plot2["trip_duration_min"], 1)
x_line = np.linspace(plot2["trip_distance"].min(), plot2["trip_distance"].max(), 200)
ax.plot(x_line, m * x_line + b, color="#2d6a4f", linewidth=1.8,
        linestyle="--", label=f"OLS  slope={m:.2f} min/mile")
ax.legend(frameon=False)

fig.tight_layout()
save(fig, "02_distance_vs_duration.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Flag analysis
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 3: Flag analysis …")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Negative-value flag analysis", fontsize=14, fontweight="bold", y=1.01)

flag_counts = (
    df_spark
    .groupBy("flag_no_meter", "flag_overcharge")
    .count()
    .toPandas()
)
flag_counts["label"] = flag_counts.apply(
    lambda r: f"no_meter={int(r.flag_no_meter)}\novercharge={int(r.flag_overcharge)}", axis=1
)
flag_counts = flag_counts.sort_values("count", ascending=False)

ax = axes[0]
bar_colors = [ACCENT, ACCENT2, RED, "#888"][:len(flag_counts)]
bars = ax.barh(flag_counts["label"], flag_counts["count"],
               color=bar_colors, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Row count")
ax.set_title("Flag combination breakdown")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
for bar, val in zip(bars, flag_counts["count"]):
    ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=9)

ax2 = axes[1]
groups = [
    pdf[pdf["flag_overcharge"] == 0]["trip_duration_min"].dropna(),
    pdf[pdf["flag_overcharge"] == 1]["trip_duration_min"].dropna(),
]
bp = ax2.boxplot(groups, patch_artist=True, notch=True,
                 medianprops=dict(color="white", linewidth=2))
for patch, color in zip(bp["boxes"], [ACCENT, RED]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax2.set_xticklabels(["No overcharge flag\n(flag=0)", "Overcharge flagged\n(flag=1)"])
ax2.set_ylabel("Duration (minutes)")
ax2.set_title("Duration distribution by overcharge flag")

fig.tight_layout()
save(fig, "03_flag_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Feature correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 4: Correlation heatmap …")

CORR_COLS = [
    "trip_duration_seconds", "trip_distance", "passenger_count",
    "pickup_hour", "pickup_dayofweek", "pickup_month",
    "is_weekend", "is_rush_hour", "store_fwd_flag_int",
    "ratecode_clean", "speed_mph_proxy",
    "flag_overcharge", "flag_no_meter",
]
present = [c for c in CORR_COLS if c in pdf.columns]
# FIX: coerce all to float — LongType / IntegerType columns may arrive as
#      int64 which is fine, but explicit cast prevents any object-dtype issues
corr_pdf = pdf[present].apply(lambda col: col.astype(float))
corr_matrix = corr_pdf.corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, vmin=-1, vmax=1,
            linewidths=0.4, linecolor="white",
            annot_kws={"size": 8}, ax=ax)
ax.set_title("Pearson correlation — cleaned features",
             fontsize=14, fontweight="bold", pad=12)
fig.tight_layout()
save(fig, "04_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Temporal patterns
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 5: Temporal patterns …")

DAY_LABELS   = {1:"Sun",2:"Mon",3:"Tue",4:"Wed",5:"Thu",6:"Fri",7:"Sat"}
MONTH_LABELS = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("Mean trip duration across time dimensions",
             fontsize=14, fontweight="bold", y=1.01)

# 5a — by hour
hour_means = pdf.groupby("pickup_hour")["trip_duration_min"].mean().reset_index()
ax = axes[0]
ax.bar(hour_means["pickup_hour"], hour_means["trip_duration_min"],
       color=ACCENT, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Pickup hour (0–23)")
ax.set_ylabel("Mean duration (min)")
ax.set_title("By hour of day")
ax.set_xticks(range(0, 24, 3))
for band in [(7, 9), (16, 19)]:
    ax.axvspan(band[0], band[1], alpha=0.12, color=RED,
               label="Rush hour" if band[0] == 7 else "")
ax.legend(frameon=False, fontsize=9)

# 5b — by day of week
dow_means = (pdf.groupby("pickup_dayofweek")["trip_duration_min"]
               .mean().reset_index().sort_values("pickup_dayofweek"))
dow_means["day_label"] = dow_means["pickup_dayofweek"].map(DAY_LABELS)
colors_dow = [RED if d in [1, 7] else ACCENT for d in dow_means["pickup_dayofweek"]]
axes[1].bar(dow_means["day_label"], dow_means["trip_duration_min"],
            color=colors_dow, edgecolor="white", linewidth=0.4)
axes[1].set_xlabel("Day of week")
axes[1].set_ylabel("Mean duration (min)")
axes[1].set_title("By day of week")
axes[1].legend(
    handles=[mpatches.Patch(facecolor=RED,    label="Weekend"),
             mpatches.Patch(facecolor=ACCENT, label="Weekday")],
    frameon=False, fontsize=9
)

# 5c — by month
month_means = (pdf.groupby("pickup_month")["trip_duration_min"]
                  .mean().reset_index().sort_values("pickup_month"))
month_means["month_label"] = month_means["pickup_month"].map(MONTH_LABELS)
axes[2].plot(month_means["month_label"], month_means["trip_duration_min"],
             marker="o", color=ACCENT2, linewidth=2, markersize=7)
axes[2].fill_between(month_means["month_label"], month_means["trip_duration_min"],
                     alpha=0.15, color=ACCENT2)
axes[2].set_xlabel("Month")
axes[2].set_ylabel("Mean duration (min)")
axes[2].set_title("By month (2025)")
axes[2].tick_params(axis="x", rotation=45)

fig.tight_layout()
save(fig, "05_temporal_patterns.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — Speed proxy distribution
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 6: Speed proxy distribution …")

SPEED_CAP  = 60.0
speed_data = pdf["speed_mph_proxy"].dropna()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle("Speed proxy distribution (miles per hour)",
             fontsize=14, fontweight="bold", y=1.01)

ax = axes[0]
sns.histplot(speed_data, bins=100, kde=True, color=ACCENT, ax=ax,
             line_kws={"linewidth": 2})
ax.axvline(SPEED_CAP, color=RED, linestyle="--", linewidth=1.5,
           label=f"Cap at {SPEED_CAP} mph")
pct_above = (speed_data > SPEED_CAP).mean() * 100
ax.text(SPEED_CAP + 1, ax.get_ylim()[1] * 0.85,
        f"{pct_above:.1f}% above cap", color=RED, fontsize=9)
ax.set_xlabel("Speed (mph)")
ax.set_ylabel("Count")
ax.set_title("Full range")
ax.legend(frameon=False)

ax2 = axes[1]
capped = speed_data[speed_data <= SPEED_CAP]
sns.histplot(capped, bins=80, kde=True, color=ACCENT2, ax=ax2,
             line_kws={"linewidth": 2})
p50_speed = capped.median()
ax2.axvline(p50_speed, color=RED, linestyle="--", linewidth=1.5,
            label=f"Median {p50_speed:.1f} mph")
ax2.set_xlabel("Speed (mph)")
ax2.set_ylabel("Count")
ax2.set_title(f"Capped at {SPEED_CAP} mph")
ax2.legend(frameon=False)

fig.tight_layout()
save(fig, "06_speed_proxy_distribution.png")

print(f"\nAll plots saved to: {PLOT_DIR}")
spark.stop()
