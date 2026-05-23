# PQ-LITS: Post-Quantum Security for Low-Altitude Intelligent Transportation Systems

This repository contains the complete implementation and experimental pipeline for the paper:

> **Quantum Safe Drone Security - Authenticated Key Exchange and Privacy Preserving Intrusion Detection for LITS**
>
> Shafiq Ahmed, Mohammad Hossein Anisi, Ayesha Iqbal, Abhirami Suresh, Mohammed Amoon, Kadambri Agarwal

## Overview

PQ-LITS is a two-layer security architecture for UAV-based low-altitude intelligent transportation systems (LITS). It combines:

1. **Post-quantum mutual authentication** using ML-KEM-768 (FIPS 203) and ML-DSA-65 (FIPS 204) at NIST Security Level 3, with dual key encapsulation, certificate-based identity binding, and cryptographic zone handover.

2. **Federated autoencoder anomaly detection** with differential privacy, enabling distributed GCS clients to collaboratively detect GPS spoofing, RF jamming, and command injection attacks without sharing raw telemetry.

## Repository Structure

```
pq-lits/
├── pq_lits_dataset_generator.py        # Synthetic UAV telemetry dataset generator
├── pq_lits_federated_training.py       # Federated learning pipeline (FedAvg + DP)
├── pq_lits_benchmark.py                # PQ-LITS protocol benchmark suite
├── requirements.txt                    # Python dependencies
├── data/
│   ├── GCS_A_telemetry.csv             # GCS-A telemetry (5000 samples, 40% GPS spoof)
│   ├── GCS_B_telemetry.csv             # GCS-B telemetry (5000 samples, 40% cmd injection)
│   ├── GCS_C_telemetry.csv             # GCS-C telemetry (5000 samples, balanced attacks)
│   └── combined_telemetry.csv          # All three GCS datasets combined
├── results/
│   ├── PQ-LITS_Protocol_Benchmark_Suite_Results
│   ├── Communication_Cost_of_PQ_LITS_
│   └── Federated_Training_Output_results_and_Tables_for_article
└── figures/
    ├── pq_lits_results.png             # Federated anomaly detection results (4 panels)
    └── pq_lits_results.pdf             # Same in vector format
```

## Key Results

### Protocol Performance (Intel i5-7360U, 2.3 GHz)

| Metric | Value |
|--------|-------|
| Full handshake latency | 0.815 ms |
| UAV-side computation | 0.410 ms |
| GCS-side computation | 0.404 ms |
| Handshake bandwidth | 15,656 bytes (3 messages) |
| Speedup over QSAKE | 50.5% faster |
| NIST Security Level | Level 3 (vs QSAKE Level 1) |

### Federated Anomaly Detection

| Approach | Accuracy | Precision | Recall | F1-Score | AUC |
|----------|----------|-----------|--------|----------|-----|
| Local-Only | 0.7488 | 0.8623 | 0.7014 | 0.7715 | 0.8483 |
| Federated | 0.7304 | 0.8441 | 0.6862 | 0.7549 | 0.8368 |
| Centralised | 0.7632 | 0.8787 | 0.7155 | 0.7888 | 0.8535 |

The federated model achieves 95.7% of the centralised F1 upper bound while preserving data locality. RF jamming detection at GCS-C improves from 52.0% (local) to 68.9% (federated), a gain of 16.9 percentage points through cross-client knowledge transfer.

## Installation

### Prerequisites

- Python 3.10 or later
- macOS, Linux, or WSL on Windows
- For protocol benchmarks: liboqs C library (built from source)

### Step 1: Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install liboqs (for protocol benchmarks only)

The protocol benchmark requires the Open Quantum Safe liboqs library. Follow the [official build instructions](https://github.com/open-quantum-safe/liboqs):

```bash
# Clone and build liboqs
git clone --branch 0.15.0 https://github.com/open-quantum-safe/liboqs.git
cd liboqs
mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(nproc)
sudo make install

# Install Python bindings
pip install liboqs-python==0.14.1
```

On macOS, if you encounter library loading errors:
```bash
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
```

On Linux:
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
sudo ldconfig
```

## Usage

### 1. Generate Synthetic Telemetry Dataset

```bash
python pq_lits_dataset_generator.py
```

This creates three non-IID telemetry CSV files (5000 samples each, 12 features, 4 classes) simulating realistic UAV flight data across three GCS sectors. The pre-generated datasets are already included in `data/`.

### 2. Run Federated Training Pipeline

```bash
python pq_lits_federated_training.py
```

This runs the complete federated learning experiment:
- Trains local autoencoders at each GCS client
- Runs FedAvg aggregation for 20 rounds with 8 local epochs
- Applies Gaussian differential privacy (sigma = 0.01)
- Evaluates local-only, federated, and centralised baselines
- Generates the 4-panel results figure

**Note:** For GPU-accelerated training, run on Google Colab with a T4 GPU and CUDA 12.8.

### 3. Run Protocol Benchmark Suite

```bash
python pq_lits_benchmark.py
```

This benchmarks all cryptographic operations in the PQ-LITS handshake:
- Per-operation latency (KEM, DSA, SHA3, HKDF, AES-GCM) over 1000 iterations
- Full handshake timing (Phases 4a, 4b, key confirmation)
- Message size analysis
- Communication cost breakdown

Requires liboqs to be installed (see Step 2 above).

## Dataset Description

Each GCS telemetry file contains 5000 samples with 18 columns:

| Feature | Description |
|---------|-------------|
| latitude, longitude, altitude | GPS coordinates and flight altitude |
| ground_speed, vertical_speed | Velocity components |
| heading | Compass bearing (degrees) |
| rssi | Received Signal Strength Indicator (dBm) |
| snr | Signal-to-Noise Ratio (dB) |
| packet_interval | Time between received packets (ms) |
| num_satellites | GPS satellite count |
| battery_voltage | Battery level (V) |
| cmd_rate | Command reception rate (Hz) |
| label | 0 = normal, 1 = GPS spoofing, 2 = RF jamming, 3 = command injection |
| attack_name | normal, gps_spoofing, rf_jamming, cmd_injection |

The non-IID distribution across clients:
- **GCS-A:** 50% normal, 40% GPS spoofing, 10% RF jamming, 0% command injection
- **GCS-B:** 50% normal, 10% GPS spoofing, 0% RF jamming, 40% command injection
- **GCS-C:** 55% normal, 15% GPS spoofing, 15% RF jamming, 15% command injection

## Cryptographic Primitives

| Primitive | Standard | Library | Parameters |
|-----------|----------|---------|------------|
| ML-KEM-768 | FIPS 203 | liboqs 0.15.0 | n=256, k=3, q=3329 |
| ML-DSA-65 | FIPS 204 | liboqs 0.15.0 | NIST Level 3 |
| SHA3-256 | FIPS 202 | Python hashlib | 256-bit digest |
| HKDF-SHA3-256 | RFC 5869 | cryptography | Extract-then-expand |
| AES-256-GCM | SP 800-38D | cryptography | 256-bit key, 96-bit nonce |

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{ahmed2026pqlits,
  title={Quantum Safe Drone Security - Authenticated Key Exchange and
         Privacy Preserving Intrusion Detection for LITS},
  author={Ahmed, Shafiq and Anisi, Mohammad Hossein and Iqbal, Ayesha 
          and Suresh, Abhirami and Amoon, Mohammed and Agarwal, Kadambri},
  journal={},
  year={2026}
}
```

## Acknowledgements

This work was supported by the Ongoing Research Funding Program (ORF-2026-968), King Saud University, Riyadh, Saudi Arabia.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
