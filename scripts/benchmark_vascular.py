import os
import sys
import time
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from process_vascular_analysis import VascularAnalysisProcessor

def make_payload(segments: int = 120, length: int = 2000) -> dict:
    data = []
    base = '2025-01-01 00:00:00'
    for i in range(segments):
        seg = {
            'collectTime': base,
            'createTime': base,
            'decompressedData': (np.sin(np.linspace(0, 50, length)) + 0.1*np.random.randn(length)).astype(float).tolist()
        }
        data.append(seg)
    return {'deviceId': 'bench_device', 'processedData': data}

def run_once(parallel: bool) -> float:
    if parallel:
        os.environ['SEGMENT_PARALLEL'] = '1'
        os.environ['SEGMENT_PARALLEL_WORKERS'] = str(max(1, min(8, (os.cpu_count() or 2))))
    else:
        os.environ['SEGMENT_PARALLEL'] = '0'
    payload = make_payload()
    p = VascularAnalysisProcessor(input_dir='output', output_dir='analysis_results')
    t0 = time.perf_counter()
    _ = p.process_device_payload(payload)
    return (time.perf_counter() - t0) * 1000.0

def main():
    b1 = run_once(False)
    b2 = run_once(True)
    out = f"非并行耗时ms={round(b1)} 并行耗时ms={round(b2)} 提升比例={(1.0 - (b2/b1)) if b1>0 else 0.0:.2f}"
    profiles_dir = os.path.join(BASE_DIR, 'profiles')
    try:
        os.makedirs(profiles_dir, exist_ok=True)
    except Exception:
        pass
    fp = os.path.join(profiles_dir, 'bench_vascular.txt')
    try:
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(out)
    except Exception:
        pass

if __name__ == '__main__':
    main()