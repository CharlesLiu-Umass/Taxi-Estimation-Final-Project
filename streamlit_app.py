"""
Streamlit UI for NYC Taxi Trip Duration Prediction.

Two tabs:
  1. Single-trip prediction – select features and get estimated duration.
  2. Load testing – simulate N concurrent users hitting the API and visualise
     latency / throughput metrics with dynamic charts.
"""

import time
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Config ───────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

DAY_NAMES = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
             5: "Thursday", 6: "Friday", 7: "Saturday"}
PAYMENT_TYPES = {1: "Credit Card", 2: "Cash", 3: "No Charge", 4: "Dispute", 5: "Unknown"}
MONTH_NAMES = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May",
               6: "June", 7: "July", 8: "August", 9: "September", 10: "October",
               11: "November", 12: "December"}

# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NYC Taxi Trip Predictor", layout="wide")
st.title("NYC Taxi Trip Duration Predictor")

# ── Sidebar: Model selector ─────────────────────────────────────────────────
MODEL_OPTIONS = {
    "Full NN (Keras)": "keras",
    "Pruned NN (TFLite)": "tflite",
}
with st.sidebar:
    st.header("Settings")
    model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=1)
    selected_model = MODEL_OPTIONS[model_label]
    st.caption(f"API query param: `model_type={selected_model}`")

tab_predict, tab_loadtest = st.tabs(["Single Trip Prediction", "Load Testing"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE TRIP PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("Enter trip details")

    col1, col2, col3 = st.columns(3)

    with col1:
        passenger_count = st.slider("Passengers", 0, 9, 1)
        trip_distance = st.slider("Trip distance (miles)", 0.01, 12.0, 2.5, step=0.01)
        payment_type_label = st.selectbox("Payment type", list(PAYMENT_TYPES.values()))
        payment_type = [k for k, v in PAYMENT_TYPES.items() if v == payment_type_label][0]

    with col2:
        pickup_hour = st.slider("Pickup hour (0–23)", 0, 23, 14)
        day_label = st.selectbox("Day of week", list(DAY_NAMES.values()))
        pickup_dayofweek = [k for k, v in DAY_NAMES.items() if v == day_label][0]

    with col3:
        month_label = st.selectbox("Pickup month", list(MONTH_NAMES.values()), index=4)
        pickup_month = [k for k, v in MONTH_NAMES.items() if v == month_label][0]
        is_weekend = 1 if pickup_dayofweek in (1, 7) else 0
        st.info(f"Weekend: {'Yes' if is_weekend else 'No'}")

    if st.button("Predict Trip Duration", use_container_width=True):
        payload = {
            "passenger_count": passenger_count,
            "trip_distance": trip_distance,
            "payment_type": payment_type,
            "pickup_hour": pickup_hour,
            "pickup_dayofweek": pickup_dayofweek,
            "pickup_month": pickup_month,
            "is_weekend": is_weekend,
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, params={"model_type": selected_model}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            secs = data["trip_duration_seconds"]
            mins = int(secs // 60)
            rem_secs = int(secs % 60)

            st.success(f"**Estimated trip duration: {mins} min {rem_secs} sec** ({secs:.1f} seconds)")

            metric_cols = st.columns(3)
            metric_cols[0].metric("Duration (seconds)", f"{secs:.1f}")
            metric_cols[1].metric("Duration (minutes)", f"{data['trip_duration_minutes']:.2f}")
            metric_cols[2].metric("Model Used", data.get("model_used", selected_model))
        except requests.ConnectionError:
            st.error("Cannot connect to the API. Make sure the FastAPI server is running on port 8000.")
        except Exception as e:
            st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LOAD TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def _random_payload() -> dict:
    """Generate a random but realistic trip payload."""
    dow = random.randint(1, 7)
    return {
        "passenger_count": random.randint(0, 6),
        "trip_distance": round(random.uniform(0.1, 11.0), 2),
        "payment_type": random.choice([1, 2, 3, 4]),
        "pickup_hour": random.randint(0, 23),
        "pickup_dayofweek": dow,
        "pickup_month": random.randint(1, 12),
        "is_weekend": 1 if dow in (1, 7) else 0,
    }


def _send_request(session: requests.Session, model_type: str = "tflite") -> dict:
    """Send a single prediction request. Returns timing information."""
    payload = _random_payload()
    start = time.perf_counter()
    try:
        resp = session.post(f"{API_URL}/predict", json=payload, params={"model_type": model_type}, timeout=30)
        latency = time.perf_counter() - start
        return {"latency": latency, "status": resp.status_code, "success": resp.status_code == 200}
    except Exception as e:
        latency = time.perf_counter() - start
        return {"latency": latency, "status": 0, "success": False, "error": str(e)}


with tab_loadtest:
    st.subheader("Simulate concurrent users")

    lt_col1, lt_col2 = st.columns(2)
    with lt_col1:
        num_users = st.number_input("Concurrent users", min_value=1, max_value=500, value=100, step=10)
    with lt_col2:
        reqs_per_user = st.number_input("Requests per user", min_value=1, max_value=100, value=5, step=1)

    total_requests = num_users * reqs_per_user
    st.caption(f"Total requests: **{total_requests}**")

    if st.button("🏁 Run Load Test", use_container_width=True):
        # Placeholders for dynamic updates
        progress_bar = st.progress(0, text="Sending requests…")
        chart_placeholder = st.empty()
        stats_placeholder = st.empty()

        results: list[dict] = []
        completed = 0
        wall_start = time.perf_counter()

        session = requests.Session()

        with ThreadPoolExecutor(max_workers=num_users) as pool:
            futures = [pool.submit(_send_request, session, selected_model) for _ in range(total_requests)]

            for future in as_completed(futures):
                result = future.result()
                result["request_num"] = completed + 1
                result["elapsed"] = time.perf_counter() - wall_start
                results.append(result)
                completed += 1

                # Update progress
                pct = completed / total_requests
                progress_bar.progress(pct, text=f"Completed {completed}/{total_requests}")

                # Update charts every ~5% or on last request
                if completed % max(1, total_requests // 20) == 0 or completed == total_requests:
                    df = pd.DataFrame(results)

                    # ── Latency chart ──
                    fig_lat = go.Figure()
                    fig_lat.add_trace(go.Scatter(
                        x=df["request_num"], y=df["latency"] * 1000,
                        mode="lines", name="Latency",
                        line=dict(color="#636EFA", width=1),
                    ))
                    # Running average
                    if len(df) >= 5:
                        df["rolling_avg"] = df["latency"].rolling(window=max(5, len(df) // 20), min_periods=1).mean() * 1000
                        fig_lat.add_trace(go.Scatter(
                            x=df["request_num"], y=df["rolling_avg"],
                            mode="lines", name="Rolling Avg",
                            line=dict(color="#EF553B", width=2),
                        ))
                    fig_lat.update_layout(
                        title="Per-Request Latency",
                        xaxis_title="Request #", yaxis_title="Latency (ms)",
                        height=350, margin=dict(t=40, b=30),
                    )

                    # ── Throughput chart ──
                    # Compute instantaneous throughput in 1-second buckets
                    df["time_bucket"] = df["elapsed"].apply(lambda t: int(t))
                    tp_df = df.groupby("time_bucket").size().reset_index(name="requests_per_sec")

                    fig_tp = go.Figure()
                    fig_tp.add_trace(go.Bar(
                        x=tp_df["time_bucket"], y=tp_df["requests_per_sec"],
                        marker_color="#00CC96", name="Throughput",
                    ))
                    fig_tp.update_layout(
                        title="Throughput Over Time",
                        xaxis_title="Time (seconds)", yaxis_title="Requests / sec",
                        height=350, margin=dict(t=40, b=30),
                    )

                    with chart_placeholder.container():
                        c1, c2 = st.columns(2)
                        c1.plotly_chart(fig_lat, use_container_width=True, key=f"lat_{completed}")
                        c2.plotly_chart(fig_tp, use_container_width=True, key=f"tp_{completed}")

        # ── Final stats ──
        wall_time = time.perf_counter() - wall_start
        df = pd.DataFrame(results)
        latencies = df["latency"].tolist()
        success_count = df["success"].sum()
        error_count = total_requests - success_count
        sorted_lat = sorted(latencies)

        def percentile(data, p):
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (k - f) * (data[c] - data[f])

        stats = {
            "Total Requests": total_requests,
            "Successful": int(success_count),
            "Failed": int(error_count),
            "Error Rate (%)": round(error_count / total_requests * 100, 2),
            "Total Time (s)": round(wall_time, 2),
            "Throughput (req/s)": round(total_requests / wall_time, 2),
            "Mean Latency (ms)": round(statistics.mean(latencies) * 1000, 2),
            "Median Latency (ms)": round(statistics.median(latencies) * 1000, 2),
            "P95 Latency (ms)": round(percentile(sorted_lat, 95) * 1000, 2),
            "P99 Latency (ms)": round(percentile(sorted_lat, 99) * 1000, 2),
            "Min Latency (ms)": round(min(latencies) * 1000, 2),
            "Max Latency (ms)": round(max(latencies) * 1000, 2),
        }

        progress_bar.progress(1.0, text="Load test complete!")

        with stats_placeholder.container():
            st.subheader("Results Summary")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Throughput", f"{stats['Throughput (req/s)']} req/s")
            s2.metric("Mean Latency", f"{stats['Mean Latency (ms)']} ms")
            s3.metric("P95 Latency", f"{stats['P95 Latency (ms)']} ms")
            s4.metric("Error Rate", f"{stats['Error Rate (%)']}%")

            st.dataframe(pd.DataFrame([stats]).T.rename(columns={0: "Value"}), use_container_width=True)

            # Latency distribution histogram
            fig_hist = px.histogram(
                df, x=df["latency"] * 1000, nbins=50,
                title="Latency Distribution",
                labels={"x": "Latency (ms)", "y": "Count"},
                color_discrete_sequence=["#AB63FA"],
            )
            fig_hist.update_layout(height=300, margin=dict(t=40, b=30))
            st.plotly_chart(fig_hist, use_container_width=True)
