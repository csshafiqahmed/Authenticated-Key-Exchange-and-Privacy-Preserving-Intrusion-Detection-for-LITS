"""
PQ-LITS Protocol Benchmark
===========================
Complete implementation and benchmarking of the PQ-LITS
authentication protocol (Phase 4a + 4b + Key Confirmation).

Cryptographic primitives:
  - ML-KEM-768  (FIPS 203) via liboqs
  - ML-DSA-65   (FIPS 204) via liboqs
  - SHA3-256    via hashlib
  - HKDF-SHA256 via cryptography
  - AES-256-GCM via cryptography

Usage:
  uv run pq_lits_benchmark.py

Author: Shafiq Ahmed
Date:   March 2026
"""

import time
import os
import hashlib
import platform
import statistics
import sys

try:
    import oqs
except ImportError:
    print("ERROR: liboqs-python not installed.")
    print("Run: uv add liboqs-python")
    sys.exit(1)

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
except ImportError:
    print("ERROR: cryptography not installed.")
    print("Run: uv add cryptography")
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================
KEM_ALG = "ML-KEM-768"
SIG_ALG = "ML-DSA-65"
ITERATIONS = 1000
WARMUP = 50


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def sha3_256(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()


def hkdf_derive(ikm: bytes, info: bytes, length: int = 32) -> bytes:
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=None,
        info=info,
    )
    return hkdf.derive(ikm)


def aes_gcm_encrypt(key: bytes, plaintext: bytes) -> tuple:
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext, None)
    return nonce, ct


def aes_gcm_decrypt(key: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)


# ============================================================
# PQ-LITS PROTOCOL IMPLEMENTATION
# ============================================================
class PQLITSProtocol:

    def setup_keys(self):
        uav_kem = oqs.KeyEncapsulation(KEM_ALG)
        uav_ek = uav_kem.generate_keypair()
        uav_dk = uav_kem.export_secret_key()

        uav_sig = oqs.Signature(SIG_ALG)
        uav_pk_s = uav_sig.generate_keypair()
        uav_sk_s = uav_sig.export_secret_key()

        gcs_kem = oqs.KeyEncapsulation(KEM_ALG)
        gcs_ek = gcs_kem.generate_keypair()
        gcs_dk = gcs_kem.export_secret_key()

        gcs_sig = oqs.Signature(SIG_ALG)
        gcs_pk_s = gcs_sig.generate_keypair()
        gcs_sk_s = gcs_sig.export_secret_key()

        ta_sig = oqs.Signature(SIG_ALG)
        ta_pk_s = ta_sig.generate_keypair()
        ta_sk_s = ta_sig.export_secret_key()

        uid_i = os.urandom(16)
        gid_k = os.urandom(16)
        zid_k = os.urandom(8)
        tau_i = sha3_256(uid_i + gid_k + zid_k)
        tau_k = sha3_256(gid_k + uid_i + zid_k)

        cert_i_data = uid_i + uav_pk_s + uav_ek + zid_k + tau_i
        cert_i = ta_sig.sign(cert_i_data)

        ta_sig2 = oqs.Signature(SIG_ALG, ta_sk_s)
        cert_k_data = gid_k + gcs_pk_s + gcs_ek + zid_k + tau_k
        cert_k = ta_sig2.sign(cert_k_data)

        return {
            'uid_i': uid_i, 'uav_ek': uav_ek, 'uav_dk': uav_dk,
            'uav_pk_s': uav_pk_s, 'uav_sk_s': uav_sk_s,
            'cert_i': cert_i, 'cert_i_data': cert_i_data, 'tau_i': tau_i,
            'gid_k': gid_k, 'gcs_ek': gcs_ek, 'gcs_dk': gcs_dk,
            'gcs_pk_s': gcs_pk_s, 'gcs_sk_s': gcs_sk_s,
            'cert_k': cert_k, 'cert_k_data': cert_k_data, 'tau_k': tau_k,
            'zid_k': zid_k, 'ta_pk_s': ta_pk_s, 'ta_sk_s': ta_sk_s,
        }

    def phase4a_uav_send(self, keys):
        n_i = os.urandom(32)
        kem_enc = oqs.KeyEncapsulation(KEM_ALG)
        ct_1, ss_1 = kem_enc.encap_secret(keys['gcs_ek'])
        ts_1 = int(time.time()).to_bytes(8, 'big')
        a_1 = sha3_256(keys['uid_i'] + n_i + ss_1 + ts_1 + keys['zid_k'] + keys['tau_i'])
        sig_uav = oqs.Signature(SIG_ALG, keys['uav_sk_s'])
        sigma_1 = sig_uav.sign(ct_1 + a_1 + ts_1 + keys['zid_k'])
        m1 = {'uid_i': keys['uid_i'], 'ct_1': ct_1, 'n_i': n_i,
              'a_1': a_1, 'sigma_1': sigma_1, 'cert_i': keys['cert_i'],
              'cert_i_data': keys['cert_i_data'], 'ts_1': ts_1}
        return m1, {'ss_1': ss_1, 'n_i': n_i}

    def phase4a_gcs_verify(self, keys, m1):
        ta_v = oqs.Signature(SIG_ALG)
        ta_v.verify(m1['cert_i_data'], m1['cert_i'], keys['ta_pk_s'])
        uav_v = oqs.Signature(SIG_ALG)
        uav_v.verify(m1['ct_1'] + m1['a_1'] + m1['ts_1'] + keys['zid_k'],
                     m1['sigma_1'], keys['uav_pk_s'])
        kem_dec = oqs.KeyEncapsulation(KEM_ALG, keys['gcs_dk'])
        ss_1 = kem_dec.decap_secret(m1['ct_1'])
        a_1_check = sha3_256(m1['uid_i'] + m1['n_i'] + ss_1 + m1['ts_1'] +
                             keys['zid_k'] + keys['tau_i'])
        assert a_1_check == m1['a_1']
        return {'ss_1': ss_1}

    def phase4b_gcs_send(self, keys, m1, gcs_state):
        n_k = os.urandom(32)
        kem_enc = oqs.KeyEncapsulation(KEM_ALG)
        ct_2, ss_2 = kem_enc.encap_secret(keys['uav_ek'])
        ikm = gcs_state['ss_1'] + ss_2
        info = m1['n_i'] + n_k + keys['uid_i'] + keys['gid_k'] + keys['zid_k']
        sk = hkdf_derive(ikm, info, 32)
        ts_2 = int(time.time()).to_bytes(8, 'big')
        a_2 = sha3_256(keys['gid_k'] + n_k + ss_2 + sk + ts_2 + keys['zid_k'])
        sig_gcs = oqs.Signature(SIG_ALG, keys['gcs_sk_s'])
        sigma_2 = sig_gcs.sign(ct_2 + a_2 + ts_2 + keys['zid_k'])
        m2 = {'gid_k': keys['gid_k'], 'ct_2': ct_2, 'n_k': n_k,
              'a_2': a_2, 'sigma_2': sigma_2, 'cert_k': keys['cert_k'],
              'cert_k_data': keys['cert_k_data'], 'ts_2': ts_2}
        gcs_state.update({'ss_2': ss_2, 'sk': sk, 'n_k': n_k})
        return m2, gcs_state

    def phase4b_uav_verify(self, keys, m1, m2, uav_state):
        ta_v = oqs.Signature(SIG_ALG)
        ta_v.verify(m2['cert_k_data'], m2['cert_k'], keys['ta_pk_s'])
        gcs_v = oqs.Signature(SIG_ALG)
        gcs_v.verify(m2['ct_2'] + m2['a_2'] + m2['ts_2'] + keys['zid_k'],
                     m2['sigma_2'], keys['gcs_pk_s'])
        kem_dec = oqs.KeyEncapsulation(KEM_ALG, keys['uav_dk'])
        ss_2 = kem_dec.decap_secret(m2['ct_2'])
        ikm = uav_state['ss_1'] + ss_2
        info = uav_state['n_i'] + m2['n_k'] + keys['uid_i'] + keys['gid_k'] + keys['zid_k']
        sk = hkdf_derive(ikm, info, 32)
        a_2_check = sha3_256(keys['gid_k'] + m2['n_k'] + ss_2 + sk + m2['ts_2'] + keys['zid_k'])
        assert a_2_check == m2['a_2']
        uav_state.update({'ss_2': ss_2, 'sk': sk})
        return uav_state

    def phase4b_key_confirm(self, keys, m2, uav_state):
        a_3 = sha3_256(uav_state['sk'] + uav_state['n_i'] + m2['n_k'] +
                       keys['uid_i'] + keys['gid_k'] + b"KC")
        ts_3 = int(time.time()).to_bytes(8, 'big')
        nonce, c_3 = aes_gcm_encrypt(uav_state['sk'], a_3 + ts_3)
        return {'nonce': nonce, 'c_3': c_3}

    def phase4b_gcs_confirm(self, keys, m2, m3, gcs_state, n_i):
        pt = aes_gcm_decrypt(gcs_state['sk'], m3['nonce'], m3['c_3'])
        a_3_received = pt[:32]
        a_3_check = sha3_256(gcs_state['sk'] + n_i + gcs_state['n_k'] +
                            keys['uid_i'] + keys['gid_k'] + b"KC")
        assert a_3_received == a_3_check
        return gcs_state['sk']


# ============================================================
# INDIVIDUAL OPERATION BENCHMARKS
# ============================================================
def benchmark_individual_operations(iterations=ITERATIONS):
    print(f"\nBenchmarking individual operations ({iterations} iterations)...")
    results = {}

    # ML-KEM-768 KeyGen
    timings = []
    for i in range(WARMUP + iterations):
        kem = oqs.KeyEncapsulation(KEM_ALG)
        t0 = time.perf_counter()
        pk = kem.generate_keypair()
        t1 = time.perf_counter()
        if i >= WARMUP:
            timings.append(t1 - t0)
    results['KEM.KeyGen'] = timings
    sk = kem.export_secret_key()
    print(f"  KEM.KeyGen:   {statistics.mean(timings)*1000:.4f} ms")

    # ML-KEM-768 Encapsulation
    timings = []
    for i in range(WARMUP + iterations):
        kem_enc = oqs.KeyEncapsulation(KEM_ALG)
        t0 = time.perf_counter()
        ct, ss = kem_enc.encap_secret(pk)
        t1 = time.perf_counter()
        if i >= WARMUP:
            timings.append(t1 - t0)
    results['KEM.Enc'] = timings
    print(f"  KEM.Enc:      {statistics.mean(timings)*1000:.4f} ms")

    # ML-KEM-768 Decapsulation
    timings = []
    for i in range(WARMUP + iterations):
        kem_dec = oqs.KeyEncapsulation(KEM_ALG, sk)
        t0 = time.perf_counter()
        ss_dec = kem_dec.decap_secret(ct)
        t1 = time.perf_counter()
        if i >= WARMUP:
            timings.append(t1 - t0)
    results['KEM.Dec'] = timings
    print(f"  KEM.Dec:      {statistics.mean(timings)*1000:.4f} ms")

    # ML-DSA-65 KeyGen
    timings = []
    for i in range(WARMUP + iterations):
        sig = oqs.Signature(SIG_ALG)
        t0 = time.perf_counter()
        pk_s = sig.generate_keypair()
        t1 = time.perf_counter()
        if i >= WARMUP:
            timings.append(t1 - t0)
    results['DSA.KeyGen'] = timings
    sk_s = sig.export_secret_key()
    print(f"  DSA.KeyGen:   {statistics.mean(timings)*1000:.4f} ms")

    # ML-DSA-65 Sign
    message = os.urandom(2048)
    timings = []
    for i in range(WARMUP + iterations):
        signer = oqs.Signature(SIG_ALG, sk_s)
        t0 = time.perf_counter()
        signature = signer.sign(message)
        t1 = time.perf_counter()
        if i >= WARMUP:
            timings.append(t1 - t0)
    results['DSA.Sign'] = timings
    print(f"  DSA.Sign:     {statistics.mean(timings)*1000:.4f} ms")

    # ML-DSA-65 Verify
    timings = []
    for i in range(WARMUP + iterations):
        verifier = oqs.Signature(SIG_ALG)
        t0 = time.perf_counter()
        valid = verifier.verify(message, signature, pk_s)
        t1 = time.perf_counter()
        if i >= WARMUP:
            timings.append(t1 - t0)
    results['DSA.Verify'] = timings
    print(f"  DSA.Verify:   {statistics.mean(timings)*1000:.4f} ms")

    # SHA3-256
    data = os.urandom(512)
    timings = []
    for i in range(WARMUP + iterations):
        t0 = time.perf_counter()
        h = sha3_256(data)
        t1 = time.perf_counter()
        if i >= WARMUP:
            timings.append(t1 - t0)
    results['SHA3-256'] = timings
    print(f"  SHA3-256:     {statistics.mean(timings)*1000:.6f} ms")

    # HKDF
    ikm = os.urandom(64)
    info = os.urandom(128)
    timings = []
    for i in range(WARMUP + iterations):
        t0 = time.perf_counter()
        key = hkdf_derive(ikm, info, 32)
        t1 = time.perf_counter()
        if i >= WARMUP:
            timings.append(t1 - t0)
    results['HKDF'] = timings
    print(f"  HKDF:         {statistics.mean(timings)*1000:.6f} ms")

    # AES-256-GCM Encrypt
    aes_key = os.urandom(32)
    plaintext = os.urandom(1024)
    timings = []
    for i in range(WARMUP + iterations):
        t0 = time.perf_counter()
        nonce, ct_aes = aes_gcm_encrypt(aes_key, plaintext)
        t1 = time.perf_counter()
        if i >= WARMUP:
            timings.append(t1 - t0)
    results['AES-GCM.Enc'] = timings
    print(f"  AES-GCM.Enc:  {statistics.mean(timings)*1000:.6f} ms")

    # AES-256-GCM Decrypt
    timings = []
    for i in range(WARMUP + iterations):
        t0 = time.perf_counter()
        pt = aes_gcm_decrypt(aes_key, nonce, ct_aes)
        t1 = time.perf_counter()
        if i >= WARMUP:
            timings.append(t1 - t0)
    results['AES-GCM.Dec'] = timings
    print(f"  AES-GCM.Dec:  {statistics.mean(timings)*1000:.6f} ms")

    return results


# ============================================================
# FULL HANDSHAKE BENCHMARK
# ============================================================
def benchmark_full_handshake(iterations=ITERATIONS):
    print(f"\nBenchmarking full handshake ({iterations} iterations)...")
    protocol = PQLITSProtocol()

    print("  Setting up keys (one-time registration)...")
    keys = protocol.setup_keys()

    print(f"  Warming up ({WARMUP} iterations)...")
    for _ in range(WARMUP):
        m1, uav_st = protocol.phase4a_uav_send(keys)
        gcs_st = protocol.phase4a_gcs_verify(keys, m1)
        m2, gcs_st = protocol.phase4b_gcs_send(keys, m1, gcs_st)
        uav_st = protocol.phase4b_uav_verify(keys, m1, m2, uav_st)
        m3 = protocol.phase4b_key_confirm(keys, m2, uav_st)
        protocol.phase4b_gcs_confirm(keys, m2, m3, gcs_st, uav_st['n_i'])

    phase_times = {
        'UAV: Construct M1 (KEM.Enc + DSA.Sign)': [],
        'GCS: Verify M1 (DSA.Vfy x2 + KEM.Dec)': [],
        'GCS: Construct M2 (KEM.Enc + DSA.Sign)': [],
        'UAV: Verify M2 (DSA.Vfy x2 + KEM.Dec)': [],
        'UAV: Key Confirm M3 (AEnc)': [],
        'GCS: Verify M3 (ADec)': [],
        'Total Handshake (M1+M2+M3)': [],
    }

    print(f"  Running {iterations} iterations...")
    for i in range(iterations):
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        m1, uav_st = protocol.phase4a_uav_send(keys)
        phase_times['UAV: Construct M1 (KEM.Enc + DSA.Sign)'].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        gcs_st = protocol.phase4a_gcs_verify(keys, m1)
        phase_times['GCS: Verify M1 (DSA.Vfy x2 + KEM.Dec)'].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        m2, gcs_st = protocol.phase4b_gcs_send(keys, m1, gcs_st)
        phase_times['GCS: Construct M2 (KEM.Enc + DSA.Sign)'].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        uav_st = protocol.phase4b_uav_verify(keys, m1, m2, uav_st)
        phase_times['UAV: Verify M2 (DSA.Vfy x2 + KEM.Dec)'].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        m3 = protocol.phase4b_key_confirm(keys, m2, uav_st)
        phase_times['UAV: Key Confirm M3 (AEnc)'].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        protocol.phase4b_gcs_confirm(keys, m2, m3, gcs_st, uav_st['n_i'])
        phase_times['GCS: Verify M3 (ADec)'].append(time.perf_counter() - t0)

        phase_times['Total Handshake (M1+M2+M3)'].append(time.perf_counter() - t_total)

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{iterations} done...")

    return phase_times


# ============================================================
# MESSAGE SIZES
# ============================================================
def measure_message_sizes():
    protocol = PQLITSProtocol()
    keys = protocol.setup_keys()
    m1, uav_st = protocol.phase4a_uav_send(keys)
    gcs_st = protocol.phase4a_gcs_verify(keys, m1)
    m2, gcs_st = protocol.phase4b_gcs_send(keys, m1, gcs_st)
    uav_st = protocol.phase4b_uav_verify(keys, m1, m2, uav_st)
    m3 = protocol.phase4b_key_confirm(keys, m2, uav_st)

    m1_size = sum(len(m1[k]) for k in ['uid_i','ct_1','n_i','a_1','sigma_1','cert_i']) + 8
    m2_size = sum(len(m2[k]) for k in ['gid_k','ct_2','n_k','a_2','sigma_2','cert_k']) + 8
    m3_size = len(m3['nonce']) + len(m3['c_3'])

    return {
        'M1': m1_size, 'M2': m2_size, 'M3': m3_size,
        'Total': m1_size + m2_size + m3_size,
        'ML-KEM-768 ek': len(keys['uav_ek']),
        'ML-KEM-768 ct': len(m1['ct_1']),
        'ML-DSA-65 vk': len(keys['uav_pk_s']),
        'ML-DSA-65 sig': len(m1['sigma_1']),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 65)
    print("PQ-LITS Protocol Benchmark Suite")
    print("=" * 65)

    print(f"\nPlatform:   {platform.platform()}")
    print(f"Processor:  {platform.processor()}")
    print(f"Machine:    {platform.machine()}")
    print(f"Python:     {platform.python_version()}")
    try:
        print(f"liboqs:     {oqs.__version__}")
    except AttributeError:
        print(f"liboqs:     (version not exposed)")
    print(f"KEM:        {KEM_ALG}")
    print(f"DSA:        {SIG_ALG}")
    print(f"Iterations: {ITERATIONS}")

    ind_results = benchmark_individual_operations()
    hs_results = benchmark_full_handshake()
    sizes = measure_message_sizes()

    # === Results Tables ===
    print("\n" + "=" * 65)
    print("INDIVIDUAL OPERATION RESULTS")
    print("=" * 65)
    print(f"{'Operation':<20} {'Mean (ms)':>12} {'Std (ms)':>12} {'Min (ms)':>12} {'Max (ms)':>12}")
    print("-" * 70)
    for op, timings in ind_results.items():
        ms = [t * 1000 for t in timings]
        print(f"{op:<20} {statistics.mean(ms):>12.4f} {statistics.stdev(ms):>12.4f} "
              f"{min(ms):>12.4f} {max(ms):>12.4f}")

    print("\n" + "=" * 65)
    print("FULL HANDSHAKE RESULTS")
    print("=" * 65)
    print(f"{'Phase':<45} {'Mean (ms)':>10} {'Std (ms)':>10}")
    print("-" * 67)
    for phase, timings in hs_results.items():
        ms = [t * 1000 for t in timings]
        marker = " <<<" if "Total" in phase else ""
        print(f"{phase:<45} {statistics.mean(ms):>10.4f} {statistics.stdev(ms):>10.4f}{marker}")

    uav_phases = ['UAV: Construct M1 (KEM.Enc + DSA.Sign)',
                  'UAV: Verify M2 (DSA.Vfy x2 + KEM.Dec)',
                  'UAV: Key Confirm M3 (AEnc)']
    gcs_phases = ['GCS: Verify M1 (DSA.Vfy x2 + KEM.Dec)',
                  'GCS: Construct M2 (KEM.Enc + DSA.Sign)',
                  'GCS: Verify M3 (ADec)']
    uav_total = sum(statistics.mean(hs_results[p]) for p in uav_phases) * 1000
    gcs_total = sum(statistics.mean(hs_results[p]) for p in gcs_phases) * 1000
    print(f"\n  UAV-side total: {uav_total:.4f} ms")
    print(f"  GCS-side total: {gcs_total:.4f} ms")

    print("\n" + "=" * 65)
    print("MESSAGE SIZES")
    print("=" * 65)
    for k, v in sizes.items():
        print(f"  {k:<20} {v:>6} bytes")

    # === LaTeX ===
    print("\n" + "=" * 65)
    print("LaTeX: Per-Operation Latency")
    print("=" * 65)
    print(r"\begin{table}[!t]")
    print(r"\centering")
    print(r"\caption{Per-Operation Cryptographic Latency (" + str(ITERATIONS) + r" iterations)}")
    print(r"\label{tab:crypto_latency}")
    print(r"\begin{tabular}{lrr}")
    print(r"\hline")
    print(r"\textbf{Operation} & \textbf{Mean (ms)} & \textbf{Std (ms)} \\")
    print(r"\hline")
    for op, timings in ind_results.items():
        ms = [t * 1000 for t in timings]
        op_tex = op.replace('_', r'\_')
        print(f"{op_tex} & {statistics.mean(ms):.4f} & {statistics.stdev(ms):.4f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\n" + "=" * 65)
    print("LaTeX: Handshake Phases")
    print("=" * 65)
    print(r"\begin{table}[!t]")
    print(r"\centering")
    print(r"\caption{PQ-LITS Handshake Latency (" + str(ITERATIONS) + r" iterations)}")
    print(r"\label{tab:handshake_latency}")
    print(r"\begin{tabular}{lrr}")
    print(r"\hline")
    print(r"\textbf{Phase} & \textbf{Mean (ms)} & \textbf{Std (ms)} \\")
    print(r"\hline")
    for phase, timings in hs_results.items():
        ms = [t * 1000 for t in timings]
        phase_tex = phase.replace('_', r'\_')
        print(f"{phase_tex} & {statistics.mean(ms):.4f} & {statistics.stdev(ms):.4f} \\\\")
    print(r"\hline")
    print(f"UAV-side total & {uav_total:.4f} & --- \\\\")
    print(f"GCS-side total & {gcs_total:.4f} & --- \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")

    total_ms = statistics.mean(hs_results['Total Handshake (M1+M2+M3)']) * 1000
    print(f"\n{'='*65}")
    print(f"COMPLETE: Full handshake = {total_ms:.2f} ms average")
    print(f"Per-packet AES-256-GCM = {statistics.mean(ind_results['AES-GCM.Enc'])*1000:.4f} ms")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
