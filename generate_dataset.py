"""
SafeDrive: Autonomous Robotics Telemetry Lifecycle & Sensor Fusion
File: generate_dataset.py

Purpose:
    Generates a realistic synthetic autonomous driving telemetry sequence
    modeled after the KITTI Vision Benchmark specifications (10 Hz sampling rate).
    Deliberately injects realistic sensor noise, physical dropouts, out-of-bound
    readings, and missing packets (~5% NaNs) to benchmark the downstream
    cleaning, feature engineering, and multi-sensor fusion pipeline.
"""

import numpy as np
import pandas as pd
import datetime

def generate_kitti_telemetry(
    num_samples: int = 400,
    sampling_rate_hz: float = 10.0,
    random_seed: int = 42,
    output_filename: str = "kitti_telemetry_raw.csv"
) -> pd.DataFrame:
    """
    Synthesizes multi-modal autonomous vehicle sensor streams at 10 Hz:
      - vehicle_speed_kmh: Speed from CAN bus / wheel encoders (km/h)
      - lidar_forward_dist_m: Front Velodyne HDL-64E forward range (meters)
      - camera_object_conf: Front stereo camera neural network detection score [0.0 - 1.0]
      - steering_angle_deg: Steering column angle in degrees [-45° to +45°]
      - imu_accel_x: Longitudinal acceleration from OXTS RT3003 GPS/IMU (m/s²)
    
    Injects realistic telemetry defects:
      - ~5% missing data (NaN dropouts)
      - Sensor dropouts / negative LiDAR distances (-3.5m, etc.)
      - Camera confidence boundary violations (> 1.0 or < 0.0)
      - Telemetry speed spikes / wheel slip anomalies (e.g. 260 km/h)
      - High-frequency sensor noise / ground jitter
    """
    np.random.seed(random_seed)
    dt = 1.0 / sampling_rate_hz
    time_seconds = np.arange(num_samples) * dt
    start_time = datetime.datetime(2026, 10, 12, 14, 30, 0, 0)
    timestamps = [start_time + datetime.timedelta(seconds=float(t)) for t in time_seconds]

    # ---------------------------------------------------------
    # Baseline Scenario Modeling (Urban driving trajectory)
    # ---------------------------------------------------------
    # Phase 1 (0 - 12s, samples 0-119): Cruising at ~48 km/h, open highway/urban road.
    # Phase 2 (12 - 24s, samples 120-239): Lead vehicle appears, our vehicle decelerates.
    # Phase 3 (24 - 32s, samples 240-319): Stop-and-wait at close proximity (6m), speed = 0 km/h.
    # Phase 4 (32 - 40s, samples 320-399): Lead vehicle drives off, vehicle accelerates back to 25 km/h.

    speed = np.zeros(num_samples)
    lidar_dist = np.zeros(num_samples)
    cam_conf = np.zeros(num_samples)
    steering = np.zeros(num_samples)
    imu_accel = np.zeros(num_samples)

    for i in range(num_samples):
        t = time_seconds[i]
        if t < 12.0:
            # Phase 1: Cruising on clear road at ~50 km/h
            speed[i] = 50.0 + 1.2 * np.sin(0.4 * t)
            lidar_dist[i] = 55.0 + 2.0 * np.cos(0.2 * t)
            cam_conf[i] = 0.05 + 0.03 * np.sin(0.5 * t)
            steering[i] = 0.5 * np.sin(0.3 * t)
            imu_accel[i] = 0.05 * np.cos(0.4 * t)
        elif t < 20.0:
            # Phase 2: Traffic detected, approaching lead vehicle
            progress = (t - 12.0) / 8.0
            speed[i] = 50.0 - (50.0 - 32.0) * progress
            lidar_dist[i] = 55.0 - (55.0 - 20.0) * progress
            cam_conf[i] = 0.08 + (0.75 - 0.08) * progress
            steering[i] = -1.5 * np.sin(np.pi * progress)
            imu_accel[i] = -1.2 + 0.1 * np.sin(t)
        elif t < 24.0:
            # Phase 3: Emergency deceleration / close proximity hazard
            progress = (t - 20.0) / 4.0
            speed[i] = 32.0 - 32.0 * progress
            lidar_dist[i] = 20.0 - (20.0 - 6.5) * progress
            cam_conf[i] = 0.75 + (0.94 - 0.75) * progress
            steering[i] = 0.2 * np.cos(t)
            imu_accel[i] = -2.8 + 0.3 * np.sin(2 * t)
        elif t < 32.0:
            # Phase 4: Stationary hold behind stopped obstacle
            speed[i] = 0.0
            lidar_dist[i] = 6.5 + 0.1 * np.sin(0.5 * t)
            cam_conf[i] = 0.94 + 0.02 * np.cos(t)
            steering[i] = 0.0
            imu_accel[i] = 0.0
        else:
            # Phase 5: Lead vehicle clears, resuming motion
            progress = (t - 32.0) / 8.0
            speed[i] = 0.0 + 28.0 * progress
            lidar_dist[i] = 6.5 + (42.0 - 6.5) * progress
            cam_conf[i] = max(0.08, 0.94 - (0.94 - 0.08) * progress)
            steering[i] = 1.0 * np.sin(0.8 * (t - 32.0))
            imu_accel[i] = 1.2 - 0.2 * progress

    # ---------------------------------------------------------
    # Add Realistic High-Frequency Sensor Jitter (Gaussian noise)
    # ---------------------------------------------------------
    speed += np.random.normal(0, 0.35, num_samples)
    lidar_dist += np.random.normal(0, 0.40, num_samples)
    cam_conf += np.random.normal(0, 0.025, num_samples)
    steering += np.random.normal(0, 0.20, num_samples)
    imu_accel += np.random.normal(0, 0.12, num_samples)

    # ---------------------------------------------------------
    # Deliberate Sensor Fault Injection (Dirty Data & Outliers)
    # ---------------------------------------------------------
    # 1. Negative LiDAR distances (sensor echo multipath / hardware dropout)
    lidar_fault_indices = [35, 112, 198, 274, 345]
    negative_lidar_values = [-3.5, -1.2, -8.0, -0.5, -4.2]
    for idx, val in zip(lidar_fault_indices, negative_lidar_values):
        lidar_dist[idx] = val

    # 2. Impossible Camera Confidence values (> 1.0 or < 0.0)
    cam_fault_indices = [45, 130, 215, 290, 370]
    invalid_cam_values = [1.45, -0.32, 2.10, 1.85, -0.15]
    for idx, val in zip(cam_fault_indices, invalid_cam_values):
        cam_conf[idx] = val

    # 3. Impossible Speed Spikes (wheel slip, CAN communication bus glitch)
    speed_spike_indices = [60, 220, 310]
    speed_spike_values = [265.0, -22.5, 288.4]
    for idx, val in zip(speed_spike_indices, speed_spike_values):
        speed[idx] = val

    # 4. Multi-Sensor Disparity Anomaly (Ghost obstacle / camera glare vs clear LiDAR)
    # Around index 75-80: Camera detects phantom obstacle (0.94 conf) while LiDAR reads clear road (46m)
    for idx in range(75, 80):
        cam_conf[idx] = 0.94 + np.random.uniform(-0.02, 0.03)

    # Assemble raw DataFrame
    df = pd.DataFrame({
        "timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] for ts in timestamps],
        "vehicle_speed_kmh": np.round(speed, 2),
        "lidar_forward_dist_m": np.round(lidar_dist, 2),
        "camera_object_conf": np.round(cam_conf, 3),
        "steering_angle_deg": np.round(steering, 2),
        "imu_accel_x": np.round(imu_accel, 2)
    })

    # ---------------------------------------------------------
    # Inject ~5% Missing Values (NaN packets) across sensor columns
    # ---------------------------------------------------------
    sensor_cols = [
        "vehicle_speed_kmh",
        "lidar_forward_dist_m",
        "camera_object_conf",
        "steering_angle_deg",
        "imu_accel_x"
    ]
    total_cells = num_samples * len(sensor_cols)
    nan_target_count = int(total_cells * 0.05)  # 5% target

    # Sample random coordinate pairs without overriding critical injected outliers
    reserved_indices = set(lidar_fault_indices + cam_fault_indices + speed_spike_indices + list(range(75, 80)))
    eligible_rows = [r for r in range(num_samples) if r not in reserved_indices]

    nan_count = 0
    while nan_count < nan_target_count:
        r = np.random.choice(eligible_rows)
        c = np.random.choice(sensor_cols)
        if not pd.isna(df.loc[r, c]):
            df.loc[r, c] = np.nan
            nan_count += 1

    # Save to disk
    df.to_csv(output_filename, index=False)
    print(f"[SafeDrive] Successfully generated {num_samples} telemetry packets at {sampling_rate_hz} Hz.")
    print(f"[SafeDrive] Raw dataset saved to: '{output_filename}'")
    print(f"[SafeDrive] Injected anomalies summary:")
    print(f"  - Total missing (NaN) cells : {df.isna().sum().sum()} (~{df.isna().sum().sum()/total_cells*100:.2f}%)")
    print(f"  - Negative LiDAR anomalies  : {sum(df['lidar_forward_dist_m'].dropna() < 0)}")
    print(f"  - Out-of-range Camera conf  : {sum((df['camera_object_conf'].dropna() < 0) | (df['camera_object_conf'].dropna() > 1))}")
    print(f"  - Impossible speed spikes   : {sum((df['vehicle_speed_kmh'].dropna() < 0) | (df['vehicle_speed_kmh'].dropna() > 160))}")

    return df

if __name__ == "__main__":
    generate_kitti_telemetry()
