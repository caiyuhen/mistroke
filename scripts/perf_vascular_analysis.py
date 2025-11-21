import os
import sys
import time
import random
import tracemalloc
import cProfile
import pstats
from datetime import datetime, timedelta
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)
from process_vascular_analysis import VascularAnalysisProcessor

def make_segment(ct: str, length: int = 2000):
    data = np.random.default_rng(42).normal(loc=0.0, scale=1.0, size=length).astype(float).tolist()
    return {"decompressedData": data, "collectTime": ct, "createTime": ct}

def make_payload(device_id: str, segments: int, length: int):
    base = datetime(2024, 8, 1, 12, 0, 0)
    arr = []
    for i in range(segments):
        ct = (base + timedelta(minutes=i*10)).strftime("%Y-%m-%d %H:%M:%S")
        arr.append(make_segment(ct, length))
    return {"deviceId": device_id, "processedData": arr}

def run_once(cfg_name: str, payload):
    tracemalloc.start()
    t0 = time.perf_counter()
    c0 = time.process_time()
    pr = cProfile.Profile()
    pr.enable()
    p = VascularAnalysisProcessor(input_dir='output', output_dir='analysis_results')
    res = p.process_device_payload(payload)
    pr.disable()
    c1 = time.process_time()
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stats = pstats.Stats(pr)
    stats.strip_dirs().sort_stats('cumulative')
    print(f"==== {cfg_name} ====")
    print(f"elapsed_sec={t1-t0:.3f} cpu_sec={c1-c0:.3f} cpu_util%={(c1-c0)/(t1-t0)*100:.1f}")
    print(f"mem_current_kb={current/1024:.1f} mem_peak_kb={peak/1024:.1f}")
    stats.print_stats(15)
    print("====================")
    return res, (t1-t0), (c1-c0), peak

def main():
    segments = int(os.environ.get('PERF_SEGMENTS', '30'))
    length = int(os.environ.get('PERF_LENGTH', '2000'))
    payload = make_payload('dev-perf', segments, length)

    os.environ.setdefault('SEGMENT_PARALLEL', '1')
    os.environ.setdefault('SEGMENT_EXECUTOR', 'thread')
    os.environ.setdefault('SEGMENT_PARALLEL_WORKERS', '8')
    os.environ.setdefault('SKIP_SLEEP_ANALYSIS', '0')

    os.environ['TS_STRING_EAGER'] = '1'
    res0, t0, c0, m0 = run_once('baseline_eager_ts', payload)

    os.environ['TS_STRING_EAGER'] = '0'
    res1, t1, c1, m1 = run_once('optimized_vectorized_ts', payload)

    print("==== summary ====")
    print(f"elapsed: baseline={t0:.3f}s optimized={t1:.3f}s improv={(t0-t1)/t0*100:.1f}%")
    print(f"cpu_sec: baseline={c0:.3f}s optimized={c1:.3f}s")
    print(f"peak_mem_kb: baseline={m0/1024:.1f} optimized={m1/1024:.1f}")

if __name__ == '__main__':
    main()