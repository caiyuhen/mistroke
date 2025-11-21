import os
import json
import time
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import iPPG_info_status as mod

def run_once(workers, batch, chunk, device_limit=None, benchmark=True):
    class Args:
        def __init__(self):
            self.profile = False
            self.benchmark = benchmark
            self.workers = workers
            self.batch_size = batch
            self.chunk_size = chunk
            self.device_limit = device_limit
    args = Args()
    t0 = time.perf_counter()
    mod.main(args)
    return round((time.perf_counter()-t0)*1000)

def main():
    baseline_ms = run_once(workers=4, batch=2000, chunk=2000, device_limit=None, benchmark=True)
    optimized_ms = run_once(workers=12, batch=6000, chunk=6000, device_limit=None, benchmark=True)
    report = {
        "baseline_ms": baseline_ms,
        "optimized_ms": optimized_ms,
        "speedup_pct": (baseline_ms - optimized_ms) * 100.0 / baseline_ms if baseline_ms > 0 else 0.0,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    out_path = os.path.join(BASE_DIR, 'logs', 'iPPG_info_status_compare.json')
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"写入对比报告失败: {e}")

if __name__ == '__main__':
    main()