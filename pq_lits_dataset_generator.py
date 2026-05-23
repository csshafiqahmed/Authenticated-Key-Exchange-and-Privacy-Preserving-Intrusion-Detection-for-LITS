"""
PQ-LITS: Synthetic UAV Telemetry Dataset Generator
====================================================
Generates realistic UAV flight telemetry data for 3 Ground Control Stations
(federated clients), each monitoring 5 UAVs across separate airspace sectors.

Attack types:
  0 = Normal flight
  1 = GPS Spoofing (gradual coordinate drift, satellite count drop)
  2 = RF Jamming (RSSI collapse, packet loss spikes)
  3 = Command Injection (anomalous heading/speed jumps)

Data is NON-IID across clients:
  - GCS-A: heavy GPS spoofing (40%), light jamming (10%)
  - GCS-B: heavy command injection (40%), light spoofing (10%)
  - GCS-C: balanced mix of all attacks (15% each)

Runtime: ~30 seconds on Colab free tier
Output:  3 CSV files (one per GCS client) + 1 combined CSV

Author: [Your Name]
Date:   March 2026
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

# ============================================================
# CONFIGURATION — Adjust these if you want more/less data
# ============================================================
SEED = 42
np.random.seed(SEED)

# Number of UAVs per GCS sector
UAVS_PER_SECTOR = 5

# Telemetry samples per UAV per flight session
# Each UAV does 2 flight sessions -> 2 * 500 = 1000 samples/UAV
SAMPLES_PER_SESSION = 500
SESSIONS_PER_UAV = 2

# Total per client: 5 UAVs * 1000 samples = 5,000
# Total dataset: 3 clients * 5,000 = 15,000 samples
# This is deliberately small. Enough for proof-of-concept, fast to train.

# Attack distribution per client (non-IID)
# Format: {attack_label: proportion}
ATTACK_DIST = {
    'GCS_A': {0: 0.50, 1: 0.40, 2: 0.10, 3: 0.00},  # GPS spoof heavy
    'GCS_B': {0: 0.50, 1: 0.10, 2: 0.00, 3: 0.40},  # Cmd injection heavy
    'GCS_C': {0: 0.55, 1: 0.15, 2: 0.15, 3: 0.15},  # Balanced mix
}

# Base coordinates for each sector (London low-altitude corridors)
SECTOR_ORIGINS = {
    'GCS_A': (51.5074, -0.1278),   # Central London
    'GCS_B': (51.4700, -0.4543),   # Heathrow vicinity
    'GCS_C': (51.5500,  0.0553),   # East London / Docklands
}

# ============================================================
# FEATURE DEFINITIONS
# ============================================================
"""
12 features per telemetry record:

1.  latitude        — GPS latitude (degrees)
2.  longitude       — GPS longitude (degrees)
3.  altitude        — Altitude above ground level in metres (20-120m for LITS)
4.  ground_speed    — Horizontal speed in m/s (0-25 for commercial UAVs)
5.  vertical_speed  — Climb/descent rate in m/s (-3 to +3)
6.  heading         — Compass heading in degrees (0-360)
7.  rssi            — Received Signal Strength Indicator in dBm (-30 to -90)
8.  snr             — Signal-to-Noise Ratio in dB (5-35)
9.  packet_interval — Time between consecutive packets in ms (50-200 normal)
10. num_satellites  — GPS satellite count (6-14 normal)
11. battery_voltage — Battery voltage in volts (10.5-12.6 for 3S LiPo)
12. cmd_rate        — Command messages received per second (1-10 normal)
"""

FEATURE_NAMES = [
    'latitude', 'longitude', 'altitude', 'ground_speed',
    'vertical_speed', 'heading', 'rssi', 'snr',
    'packet_interval', 'num_satellites', 'battery_voltage', 'cmd_rate'
]


# ============================================================
# NORMAL FLIGHT PROFILE GENERATOR
# ============================================================
def generate_normal_flight(n_samples, origin_lat, origin_lon, uav_id):
    """
    Simulate a normal UAV flight: takeoff, cruise with gentle turns,
    and landing. All features stay within physically plausible bounds.
    """
    t = np.linspace(0, 1, n_samples)  # Normalised time [0,1]

    # --- GPS coordinates: smooth path with gentle drift ---
    # UAV flies a rough circuit around the sector origin
    radius = 0.005 + np.random.uniform(0, 0.003)  # ~500m radius
    phase = np.random.uniform(0, 2 * np.pi)
    lat = origin_lat + radius * np.sin(2 * np.pi * t + phase)
    lon = origin_lon + radius * np.cos(2 * np.pi * t + phase)
    # Add small GPS noise (normal operation)
    lat += np.random.normal(0, 0.00002, n_samples)
    lon += np.random.normal(0, 0.00002, n_samples)

    # --- Altitude: takeoff -> cruise -> land ---
    cruise_alt = np.random.uniform(40, 100)
    alt = np.zeros(n_samples)
    takeoff_end = int(0.1 * n_samples)
    land_start = int(0.9 * n_samples)
    alt[:takeoff_end] = np.linspace(0, cruise_alt, takeoff_end)
    alt[takeoff_end:land_start] = cruise_alt + np.random.normal(0, 1.5, land_start - takeoff_end)
    alt[land_start:] = np.linspace(cruise_alt, 0, n_samples - land_start)
    alt = np.clip(alt, 0, 120)

    # --- Speeds ---
    cruise_speed = np.random.uniform(8, 18)
    gs = np.full(n_samples, cruise_speed)
    gs[:takeoff_end] = np.linspace(0, cruise_speed, takeoff_end)
    gs[land_start:] = np.linspace(cruise_speed, 0, n_samples - land_start)
    gs += np.random.normal(0, 0.5, n_samples)
    gs = np.clip(gs, 0, 25)

    vs = np.gradient(alt) * 2  # Approximate vertical speed
    vs = np.clip(vs, -3, 3)

    # --- Heading: smooth with gentle turns ---
    heading = np.degrees(np.arctan2(np.gradient(lon), np.gradient(lat))) % 360
    heading += np.random.normal(0, 2, n_samples)
    heading = heading % 360

    # --- RF signals: stable with minor fluctuation ---
    rssi = np.random.normal(-55, 3, n_samples)
    rssi = np.clip(rssi, -90, -30)

    snr = np.random.normal(22, 2, n_samples)
    snr = np.clip(snr, 5, 35)

    # --- Packet timing: regular ---
    pkt = np.random.normal(100, 10, n_samples)  # ~100ms interval
    pkt = np.clip(pkt, 50, 200)

    # --- GPS satellites: stable ---
    n_sat = np.random.choice([8, 9, 10, 11, 12], n_samples,
                             p=[0.1, 0.2, 0.3, 0.25, 0.15])

    # --- Battery: slow linear discharge ---
    batt = np.linspace(12.6, 11.0 + np.random.uniform(-0.3, 0.3), n_samples)
    batt += np.random.normal(0, 0.02, n_samples)
    batt = np.clip(batt, 10.5, 12.6)

    # --- Command rate: stable ---
    cmd = np.random.normal(5, 1, n_samples)
    cmd = np.clip(cmd, 1, 10)

    data = np.column_stack([lat, lon, alt, gs, vs, heading,
                            rssi, snr, pkt, n_sat, batt, cmd])
    return data


# ============================================================
# ATTACK INJECTION FUNCTIONS
# ============================================================
def inject_gps_spoofing(data, intensity='medium'):
    """
    GPS Spoofing: Attacker sends fake GPS signals.
    Observable effects:
      - Gradual coordinate drift (lat/lon shift away from true path)
      - Satellite count drops or becomes suspiciously constant
      - Altitude may become inconsistent with vertical speed
    """
    spoofed = data.copy()
    n = len(data)

    # Determine attack window (middle 60% of flight)
    start = int(0.2 * n)
    end = int(0.8 * n)
    attack_len = end - start

    # Coordinate drift: starts subtle, grows over time
    if intensity == 'strong':
        max_drift = np.random.uniform(0.005, 0.01)   # ~500m-1km drift
    else:
        max_drift = np.random.uniform(0.001, 0.005)   # ~100-500m drift

    drift_profile = np.linspace(0, max_drift, attack_len) ** 1.5  # Non-linear growth
    drift_angle = np.random.uniform(0, 2 * np.pi)

    spoofed[start:end, 0] += drift_profile * np.cos(drift_angle)  # lat
    spoofed[start:end, 1] += drift_profile * np.sin(drift_angle)  # lon

    # Satellite count: drops to 3-5 (spoofed signals overpower fewer sats)
    spoofed[start:end, 9] = np.random.choice([3, 4, 5], attack_len,
                                              p=[0.3, 0.4, 0.3])

    # Altitude inconsistency: random jumps uncorrelated with vertical speed
    alt_noise = np.random.normal(0, 8, attack_len)
    spoofed[start:end, 2] += alt_noise

    return spoofed


def inject_rf_jamming(data, intensity='medium'):
    """
    RF Jamming: Attacker floods the communication channel.
    Observable effects:
      - RSSI drops sharply (signal drowned out)
      - SNR collapses
      - Packet intervals spike (lost/delayed packets)
      - Command rate drops
    """
    jammed = data.copy()
    n = len(data)

    # Jamming in bursts (more realistic than continuous)
    n_bursts = np.random.randint(3, 7)
    for _ in range(n_bursts):
        burst_start = np.random.randint(int(0.1 * n), int(0.8 * n))
        burst_len = np.random.randint(20, 80)
        burst_end = min(burst_start + burst_len, n)

        # RSSI collapse
        jammed[burst_start:burst_end, 6] = np.random.normal(-82, 3,
                                                              burst_end - burst_start)
        # SNR collapse
        jammed[burst_start:burst_end, 7] = np.random.normal(6, 2,
                                                              burst_end - burst_start)
        # Packet interval spikes
        jammed[burst_start:burst_end, 8] = np.random.uniform(300, 1500,
                                                               burst_end - burst_start)
        # Command rate drops
        jammed[burst_start:burst_end, 11] = np.random.uniform(0, 2,
                                                                burst_end - burst_start)

    # Clip to physical bounds
    jammed[:, 6] = np.clip(jammed[:, 6], -95, -30)
    jammed[:, 7] = np.clip(jammed[:, 7], 0, 35)
    jammed[:, 8] = np.clip(jammed[:, 8], 50, 2000)
    jammed[:, 11] = np.clip(jammed[:, 11], 0, 10)

    return jammed


def inject_command_injection(data, intensity='medium'):
    """
    Command Injection: Attacker sends unauthorized flight commands.
    Observable effects:
      - Sudden heading changes (>45 deg jump in 1 timestep)
      - Speed spikes or drops not matching flight profile
      - Command rate anomaly (burst of commands)
      - Altitude sudden changes
    """
    injected = data.copy()
    n = len(data)

    # 5-10 injection events
    n_events = np.random.randint(5, 12)
    for _ in range(n_events):
        event_idx = np.random.randint(int(0.15 * n), int(0.85 * n))
        event_len = np.random.randint(5, 25)
        event_end = min(event_idx + event_len, n)

        # Heading jump: sudden 45-180 degree change
        heading_jump = np.random.uniform(45, 180) * np.random.choice([-1, 1])
        injected[event_idx:event_end, 5] = (
            injected[event_idx, 5] + heading_jump
        ) % 360

        # Speed anomaly
        injected[event_idx:event_end, 3] = np.random.uniform(0, 25,
                                                               event_end - event_idx)

        # Command rate spike (attacker flooding commands)
        injected[event_idx:event_end, 11] = np.random.uniform(15, 40,
                                                                event_end - event_idx)

        # Altitude sudden change
        alt_jump = np.random.uniform(-30, 30)
        injected[event_idx:event_end, 2] += alt_jump

    injected[:, 2] = np.clip(injected[:, 2], 0, 150)
    injected[:, 3] = np.clip(injected[:, 3], 0, 30)
    injected[:, 11] = np.clip(injected[:, 11], 0, 50)

    return injected


ATTACK_FUNCTIONS = {
    1: inject_gps_spoofing,
    2: inject_rf_jamming,
    3: inject_command_injection,
}


# ============================================================
# DATASET GENERATION
# ============================================================
def generate_client_dataset(client_name, attack_dist, origin):
    """
    Generate a complete dataset for one GCS client.
    Returns a DataFrame with features + metadata columns.
    """
    all_records = []
    origin_lat, origin_lon = origin

    for uav_idx in range(UAVS_PER_SECTOR):
        uav_id = f"{client_name}_UAV_{uav_idx:02d}"

        for session in range(SESSIONS_PER_UAV):
            # Generate base normal flight
            flight_data = generate_normal_flight(
                SAMPLES_PER_SESSION, origin_lat, origin_lon, uav_id
            )

            # Decide attack type for this session based on distribution
            attack_type = np.random.choice(
                list(attack_dist.keys()),
                p=list(attack_dist.values())
            )

            # Apply attack if not normal
            if attack_type in ATTACK_FUNCTIONS:
                intensity = np.random.choice(['medium', 'strong'],
                                             p=[0.7, 0.3])
                flight_data = ATTACK_FUNCTIONS[attack_type](
                    flight_data, intensity
                )

            # Create labels array
            # For normal flights, all samples are label 0
            # For attacks, samples in the attack window get the attack label,
            # samples outside the window remain label 0
            labels = np.zeros(SAMPLES_PER_SESSION, dtype=int)
            if attack_type != 0:
                attack_start = int(0.15 * SAMPLES_PER_SESSION)
                attack_end = int(0.85 * SAMPLES_PER_SESSION)
                labels[attack_start:attack_end] = attack_type

            # Timestamps (1 second intervals)
            base_time = datetime(2026, 3, 15, 8, 0, 0) + timedelta(
                hours=uav_idx * 3 + session
            )
            timestamps = [base_time + timedelta(seconds=i)
                          for i in range(SAMPLES_PER_SESSION)]

            # Assemble records
            for i in range(SAMPLES_PER_SESSION):
                record = {
                    'timestamp': timestamps[i].isoformat(),
                    'client_id': client_name,
                    'uav_id': uav_id,
                    'session_id': f"{uav_id}_S{session:02d}",
                }
                for j, fname in enumerate(FEATURE_NAMES):
                    record[fname] = round(flight_data[i, j], 6)
                record['label'] = int(labels[i])
                record['attack_name'] = ['normal', 'gps_spoofing',
                                         'rf_jamming', 'cmd_injection'][int(labels[i])]
                all_records.append(record)

    df = pd.DataFrame(all_records)
    return df


def main():
    print("=" * 60)
    print("PQ-LITS Synthetic UAV Telemetry Dataset Generator")
    print("=" * 60)

    output_dir = 'pq_lits_dataset'
    os.makedirs(output_dir, exist_ok=True)

    all_dfs = []

    for client_name in ['GCS_A', 'GCS_B', 'GCS_C']:
        print(f"\nGenerating data for {client_name}...")
        df = generate_client_dataset(
            client_name,
            ATTACK_DIST[client_name],
            SECTOR_ORIGINS[client_name]
        )
        all_dfs.append(df)

        # Save individual client file
        filepath = os.path.join(output_dir, f'{client_name}_telemetry.csv')
        df.to_csv(filepath, index=False)

        # Print statistics
        total = len(df)
        print(f"  Total samples:     {total}")
        for lbl, name in enumerate(['normal', 'gps_spoofing',
                                     'rf_jamming', 'cmd_injection']):
            count = (df['label'] == lbl).sum()
            pct = 100 * count / total
            print(f"  {name:20s}: {count:5d} ({pct:5.1f}%)")

    # Combined dataset
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(os.path.join(output_dir, 'combined_telemetry.csv'),
                    index=False)

    print(f"\n{'=' * 60}")
    print(f"Dataset generation complete!")
    print(f"Total samples:  {len(combined)}")
    print(f"Total features: {len(FEATURE_NAMES)}")
    print(f"Output dir:     {output_dir}/")
    print(f"Files created:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {f:35s}  ({size / 1024:.1f} KB)")
    print(f"{'=' * 60}")

    # ---- Quick sanity check: feature statistics ----
    print("\nFeature statistics (combined):")
    print(combined[FEATURE_NAMES].describe().round(4).to_string())

    # ---- Non-IID verification ----
    print("\nNon-IID verification (attack distribution per client):")
    pivot = combined.groupby(['client_id', 'attack_name']).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    print(pivot_pct.round(1).to_string())

    return combined


if __name__ == '__main__':
    combined_df = main()
