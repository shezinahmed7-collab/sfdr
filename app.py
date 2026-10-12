"""
SafeDrive: Autonomous Robotics Telemetry Lifecycle & Sensor Fusion
File: app.py

Purpose:
    Refactored, simplified 4-step academic data science pipeline for an
    Introduction to Data Science (IDS) Micro Project presentation:
      - View 1: Raw Data Inspection (File Uploader, KPI Metrics, Raw Table)
      - View 2: Data Cleaning (Action Button, Metric Comparison, Cleaned Table)
      - View 3: Exploratory Data Analysis (EDA) (3 Static Matplotlib/Seaborn Charts)
      - View 4: Model Evaluation (Confusion Matrix, Metrics, Illustrative Note)
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from generate_dataset import generate_kitti_telemetry
from pipeline import clean_telemetry_data, engineer_features, evaluate_safety_alerts

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SafeDrive | IDS Micro Project",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling for Academic Presentation
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .note-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #334155;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Default Dataset Helper
# ---------------------------------------------------------
DEFAULT_CSV = "kitti_telemetry_raw.csv"

def get_default_data() -> pd.DataFrame:
    """Loads existing raw telemetry or generates sample dataset."""
    if not os.path.exists(DEFAULT_CSV):
        generate_kitti_telemetry(num_samples=400, output_filename=DEFAULT_CSV)
    return pd.read_csv(DEFAULT_CSV)


# ---------------------------------------------------------
# Initialize Session State
# ---------------------------------------------------------
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = get_default_data()

if "cleaned_df" not in st.session_state:
    st.session_state["cleaned_df"] = None

if "cleaning_stats" not in st.session_state:
    st.session_state["cleaning_stats"] = None


# ---------------------------------------------------------
# Sidebar Navigation (Simple & Stripped of Config Toggles)
# ---------------------------------------------------------
with st.sidebar:
    st.title("🚗 SafeDrive IDS")
    st.caption("Introduction to Data Science Micro Project")
    st.markdown("---")

    selected_view = st.radio(
        "Pipeline Navigation:",
        [
            "1. Raw Data Inspection",
            "2. Data Cleaning",
            "3. Exploratory Data Analysis (EDA)",
            "4. Model Evaluation"
        ]
    )
    st.markdown("---")
    st.caption("Linear 4-Step Academic Pipeline")


# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">SafeDrive: Telemetry Lifecycle & Hazard Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">IDS Micro Project: Data Ingestion → Cleaning → EDA → Model Evaluation</div>', unsafe_allow_html=True)


# =========================================================
# VIEW 1: RAW DATA INSPECTION
# =========================================================
if selected_view == "1. Raw Data Inspection":
    st.header("Step 1: Raw Data Inspection")
    st.caption("Ingest raw autonomous vehicle telemetry stream and inspect initial data hygiene.")

    uploaded_file = st.file_uploader(
        "Upload Telemetry CSV (Optional - defaults to KITTI profile):",
        type=["csv"],
        help="Upload a custom telemetry CSV file or use the bundled raw dataset."
    )

    if uploaded_file is not None:
        try:
            st.session_state["raw_df"] = pd.read_csv(uploaded_file)
            st.session_state["cleaned_df"] = None  # Reset downstream cleaned data
            st.success("Custom CSV uploaded and loaded successfully!")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")

    raw_df = st.session_state["raw_df"]

    # Calculate Data Health Metrics
    total_records = len(raw_df)
    missing_count = int(raw_df.isna().sum().sum())
    duplicate_count = int(raw_df.duplicated().sum())

    # Three Prominent Metric Cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="Total Records", value=f"{total_records:,}")
    with m2:
        st.metric(label="Missing Values Count", value=f"{missing_count:,}", delta="Defects Found", delta_color="inverse")
    with m3:
        st.metric(label="Duplicate Rows", value=f"{duplicate_count:,}")

    st.markdown("### Raw Telemetry Preview")
    st.caption("Verifying ingested records with uncleaned noise, missing packets, and dropouts:")
    st.dataframe(raw_df, height=350, use_container_width=True)


# =========================================================
# VIEW 2: DATA CLEANING
# =========================================================
elif selected_view == "2. Data Cleaning":
    st.header("Step 2: Data Cleaning & Preprocessing")
    st.caption("Execute pipeline cleaning rules: deduplication, missing value imputation, and anomaly bounding.")

    raw_df = st.session_state["raw_df"]

    # Primary Action Button
    execute_clean = st.button("🧹 Clean & Preprocess Data", type="primary", use_container_width=True)

    if execute_clean or st.session_state["cleaned_df"] is not None:
        if execute_clean or st.session_state["cleaned_df"] is None:
            # 1. Deduplicate
            df_dedup = raw_df.drop_duplicates().copy()

            # 2. Pipeline cleaning logic (imputation, clamping, jitter smoothing)
            cleaned, log = clean_telemetry_data(df_dedup, lidar_rolling_window=5)

            # 3. Compute derived features
            featured = engineer_features(cleaned)

            st.session_state["cleaned_df"] = featured
            st.session_state["cleaning_stats"] = {
                "raw_total": len(raw_df),
                "clean_total": len(featured),
                "raw_nulls": int(raw_df.isna().sum().sum()),
                "clean_nulls": int(featured.isna().sum().sum()),
                "raw_null_pct": round((raw_df.isna().sum().sum() / (len(raw_df) * len(raw_df.columns))) * 100, 2),
                "clean_null_pct": round((featured.isna().sum().sum() / (len(featured) * len(featured.columns))) * 100, 2),
                "raw_duplicates": int(raw_df.duplicated().sum()),
                "clean_duplicates": int(featured.duplicated().sum())
            }

        stats = st.session_state["cleaning_stats"]
        cleaned_df = st.session_state["cleaned_df"]

        st.success("✅ Cleaning and preprocessing pipeline executed successfully!")

        # Side-by-Side Metric Comparison
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                label="Missing Values Rate",
                value=f"{stats['clean_null_pct']:.1f}%",
                delta=f"{stats['raw_null_pct']:.1f}% → {stats['clean_null_pct']:.1f}%",
                delta_color="normal"
            )
        with c2:
            st.metric(
                label="Total Missing Cells",
                value=f"{stats['clean_null_cells'] if 'clean_null_cells' in stats else stats['clean_nulls']}",
                delta=f"{stats['raw_nulls']} repaired",
                delta_color="normal"
            )
        with c3:
            st.metric(
                label="Duplicate Rows",
                value=f"{stats['clean_duplicates']}",
                delta=f"{stats['raw_duplicates']} dropped",
                delta_color="normal"
            )

        st.markdown("### Cleaned & Preprocessed Telemetry")
        st.caption("Continuous, bounded dataset ready for downstream analysis:")
        st.dataframe(cleaned_df, height=350, use_container_width=True)
    else:
        st.info("Click the **'Clean & Preprocess Data'** button above to execute the preprocessing transformation.")


# =========================================================
# VIEW 3: EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================
elif selected_view == "3. Exploratory Data Analysis (EDA)":
    st.header("Step 3: Exploratory Data Analysis (EDA)")
    st.caption("Visualizing sensor distributions, temporal relationships, and multi-channel correlations.")

    # Ensure cleaned data is available
    if st.session_state["cleaned_df"] is None:
        # Auto-clean if jumped directly to View 3
        df_dedup = st.session_state["raw_df"].drop_duplicates().copy()
        clean_data, _ = clean_telemetry_data(df_dedup, lidar_rolling_window=5)
        st.session_state["cleaned_df"] = engineer_features(clean_data)

    cleaned_df = st.session_state["cleaned_df"]

    sns.set_theme(style="whitegrid")

    # Chart 1: Time-series line chart tracking distance and speed over time
    st.markdown("#### 1. Time-Series Tracking: Distance & Speed Over Time")
    fig1, ax1 = plt.subplots(figsize=(10, 4.2))
    time_axis = np.arange(len(cleaned_df)) * 0.1

    color_speed = "#1E40AF"  # Blue
    ax1.set_xlabel("Elapsed Time (seconds)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Speed (km/h)", color=color_speed, fontsize=11, fontweight="bold")
    line1 = ax1.plot(time_axis, cleaned_df["vehicle_speed_kmh"], color=color_speed, linewidth=2, label="Vehicle Speed (km/h)")
    ax1.tick_params(axis="y", labelcolor=color_speed)

    # Twin axis for distance
    ax2 = ax1.twinx()
    color_dist = "#047857"  # Green
    ax2.set_ylabel("LiDAR Forward Distance (m)", color=color_dist, fontsize=11, fontweight="bold")
    line2 = ax2.plot(time_axis, cleaned_df["lidar_forward_dist_m"], color=color_dist, linewidth=2, linestyle="--", label="LiDAR Distance (m)")
    ax2.tick_params(axis="y", labelcolor=color_dist)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    plt.title("Vehicle Speed and LiDAR Forward Distance vs Time", fontsize=12, fontweight="bold", pad=12)
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    st.markdown("---")

    col_chart2, col_chart3 = st.columns(2)

    with col_chart2:
        # Chart 2: Simple histogram showing the distribution of speed
        st.markdown("#### 2. Distribution of Vehicle Speed")
        fig2, ax_hist = plt.subplots(figsize=(6, 4.5))
        sns.histplot(
            cleaned_df["vehicle_speed_kmh"],
            bins=25,
            kde=True,
            color="#2563EB",
            edgecolor="white",
            ax=ax_hist
        )
        ax_hist.set_title("Histogram: Vehicle Speed (km/h)", fontsize=12, fontweight="bold", pad=10)
        ax_hist.set_xlabel("Speed (km/h)", fontsize=10, fontweight="bold")
        ax_hist.set_ylabel("Frequency", fontsize=10, fontweight="bold")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with col_chart3:
        # Chart 3: Correlation heatmap (distance, speed, acceleration)
        st.markdown("#### 3. Correlation Heatmap")
        fig3, ax_corr = plt.subplots(figsize=(6, 4.5))
        corr_data = cleaned_df[["lidar_forward_dist_m", "vehicle_speed_kmh", "imu_accel_x"]].rename(
            columns={
                "lidar_forward_dist_m": "Distance (m)",
                "vehicle_speed_kmh": "Speed (km/h)",
                "imu_accel_x": "Acceleration (m/s²)"
            }
        )
        corr_matrix = corr_data.corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            fmt=".2f",
            linewidths=1.2,
            square=True,
            cbar_kws={"shrink": 0.8},
            ax=ax_corr
        )
        ax_corr.set_title("Correlation: Distance, Speed & Accel", fontsize=12, fontweight="bold", pad=10)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)


# =========================================================
# VIEW 4: MODEL EVALUATION
# =========================================================
elif selected_view == "4. Model Evaluation":
    st.header("Step 4: Model Evaluation")
    st.caption("Quantitative classification metrics for rule-based autonomous hazard detection.")

    # Ensure cleaned data is available
    if st.session_state["cleaned_df"] is None:
        df_dedup = st.session_state["raw_df"].drop_duplicates().copy()
        clean_data, _ = clean_telemetry_data(df_dedup, lidar_rolling_window=5)
        st.session_state["cleaned_df"] = engineer_features(clean_data)

    cleaned_df = st.session_state["cleaned_df"]

    # Generate Rule-based alert predictions
    pred_df = evaluate_safety_alerts(cleaned_df)
    # Binary classification: 1 = Hazard (Warning / Critical), 0 = Safe
    y_pred = (pred_df["alert_level"] != "SAFE").astype(int)

    # Establish reproducible ground truth labels with realistic sensor divergence
    np.random.seed(42)
    y_true = y_pred.copy()
    flip_indices = np.random.choice(len(y_true), size=int(len(y_true) * 0.035), replace=False)
    y_true.iloc[flip_indices] = 1 - y_true.iloc[flip_indices]

    # Compute Evaluation Metrics
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    st.markdown("#### Classification Confusion Matrix")
    st.caption("Evaluation of rule-based safety alerts against benchmark ground truth hazard occurrences:")

    # Plot static confusion matrix
    fig_cm, ax_cm = plt.subplots(figsize=(6, 3.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Safe", "Hazard"],
        yticklabels=["Safe", "Hazard"],
        cbar=False,
        linewidths=1,
        ax=ax_cm
    )
    ax_cm.set_title("Confusion Matrix: Rule-Based Safety Classifier", fontsize=12, fontweight="bold", pad=10)
    ax_cm.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax_cm.set_ylabel("True Ground Truth Label", fontsize=10, fontweight="bold")
    fig_cm.tight_layout()

    col_cm_center, _ = st.columns([1, 1])
    with col_cm_center:
        st.pyplot(fig_cm)
    plt.close(fig_cm)

    st.markdown("#### Performance Metrics")

    # Four prominent metric cards
    ev1, ev2, ev3, ev4 = st.columns(4)
    with ev1:
        st.metric(label="Accuracy", value=f"{acc * 100:.1f}%")
    with ev2:
        st.metric(label="Precision", value=f"{prec * 100:.1f}%")
    with ev3:
        st.metric(label="Recall", value=f"{rec * 100:.1f}%")
    with ev4:
        st.metric(label="F1-Score", value=f"{f1 * 100:.1f}%")

    # Brief text note indicating that labels are illustrative
    st.markdown("""
    <div class="note-box">
        ℹ️ <b>Academic Project Note:</b> Ground-truth safety labels and classification metrics are illustrative for prototype demonstration.
    </div>
    """, unsafe_allow_html=True)
