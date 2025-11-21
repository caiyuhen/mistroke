#!/usr/bin/env python3
import os
import sys
import json
import time
import tracemalloc
import cProfile
import pstats
from io import StringIO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)

from process_vascular_analysis import VascularAnalysisProcessor  # noqa
from ppg_inflammation_detection import PPGInflammationDetector  # noqa
from ppg_arrhythmia_detection import PPGArrhythmiaDetector  # noqa

def run_once(limit_files=3):
    proc = VascularAnalysisProcessor(
        input_dir=os.path.join(BASE_DIR, 'output'),
        output_dir=os.path.join(BASE_DIR, 'analysis_results')
    )
    # 仅处理前 N 个文件以控制基准时间
    files = [f for f in os.listdir(proc.input_dir) if f.endswith('.json')][:limit_files]
    results = []
    for fn in files:
        fp = os.path.join(proc.input_dir, fn)
        r = proc.process_device_data(fp)
        if r is not None:
            proc.save_analysis_result(r['device_id'], r)
            results.append(r)
    return len(results)

def profile_run(limit_files=3):
    tracemalloc.start()
    start = time.perf_counter()
    pr = cProfile.Profile()
    pr.enable()
    processed = run_once(limit_files)
    pr.disable()
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats(40)
    return {
        'processed_files': processed,
        'elapsed_seconds': elapsed,
        'mem_current_bytes': current,
        'mem_peak_bytes': peak,
        'stats': s.getvalue()
    }

def microbenchmarks():
    import numpy as np
    x = np.sin(np.linspace(0, 100, 50000)) + 0.1 * np.random.randn(50000)
    inflam = PPGInflammationDetector(sampling_rate=125)
    arr = PPGArrhythmiaDetector(sampling_rate=125)

    def time_fn(fn):
        t0 = time.perf_counter()
        _ = fn()
        return time.perf_counter() - t0

    # baseline
    os.environ['FAST_FD'] = '0'
    os.environ['FAST_AUTOCORR'] = '0'
    tb_fd = time_fn(lambda: inflam._calculate_waveform_complexity(x))
    tb_sqi = time_fn(lambda: arr._calculate_signal_quality(x))

    # optimized
    os.environ['FAST_FD'] = '1'
    os.environ['FAST_AUTOCORR'] = '1'
    to_fd = time_fn(lambda: inflam._calculate_waveform_complexity(x))
    to_sqi = time_fn(lambda: arr._calculate_signal_quality(x))

    return {
        'waveform_complexity': {
            'baseline': tb_fd,
            'optimized': to_fd,
            'improvement_pct': ((tb_fd - to_fd) / tb_fd * 100.0) if tb_fd > 0 else 0.0
        },
        'signal_quality_autocorr': {
            'baseline': tb_sqi,
            'optimized': to_sqi,
            'improvement_pct': ((tb_sqi - to_sqi) / tb_sqi * 100.0) if tb_sqi > 0 else 0.0
        }
    }

def main():
    os.makedirs(os.path.join(BASE_DIR, 'profiles'), exist_ok=True)
    report_path = os.path.join(BASE_DIR, 'profiles', 'perf_report.json')
    stats_path = os.path.join(BASE_DIR, 'profiles', 'perf_stats.txt')
    func_path = os.path.join(BASE_DIR, 'profiles', 'func_report.json')

    # 读取限制数量
    try:
        limit_env = int(os.environ.get('PERF_LIMIT_FILES', '2'))
    except Exception:
        limit_env = 2

    os.environ['VASCULAR_SLOW_SERIALIZE'] = '1'
    os.environ['SKIP_SLEEP_ANALYSIS'] = '1'
    os.environ['FAST_AUTOCORR'] = '0'
    os.environ['INFLAM_K_MAX'] = ''
    os.environ['INFLAM_DOWNSAMPLE'] = ''
    baseline = profile_run(limit_env)

    os.environ['VASCULAR_SLOW_SERIALIZE'] = '0'
    os.environ['SKIP_SLEEP_ANALYSIS'] = '1'
    os.environ['FAST_AUTOCORR'] = '1'
    os.environ['INFLAM_K_MAX'] = '6'
    os.environ['INFLAM_DOWNSAMPLE'] = '2'
    os.environ['FAST_FD'] = '1'
    optimized = profile_run(limit_env)

    summary = {
        'baseline': {
            'elapsed_seconds': baseline['elapsed_seconds'],
            'mem_peak_bytes': baseline['mem_peak_bytes'],
            'processed_files': baseline['processed_files']
        },
        'optimized': {
            'elapsed_seconds': optimized['elapsed_seconds'],
            'mem_peak_bytes': optimized['mem_peak_bytes'],
            'processed_files': optimized['processed_files']
        },
        'improvement_pct': (
            (baseline['elapsed_seconds'] - optimized['elapsed_seconds']) / baseline['elapsed_seconds'] * 100.0
            if baseline['elapsed_seconds'] > 0 else 0.0
        )
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write('=== Baseline cProfile ===\n')
        f.write(baseline['stats'])
        f.write('\n=== Optimized cProfile ===\n')
        f.write(optimized['stats'])

    funcs = microbenchmarks()
    with open(func_path, 'w', encoding='utf-8') as f:
        json.dump(funcs, f, indent=2, ensure_ascii=False)

    print(f"报告已写入: {report_path}")
    print(f"详细统计写入: {stats_path}")
    print(f"函数微基准写入: {func_path}")
    print(f"耗时提升: {summary['improvement_pct']:.1f}%")

if __name__ == '__main__':
    main()