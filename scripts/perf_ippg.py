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

import iPPG_info as ippg  # noqa

def run_main_once():
    ippg.main()

def profile_run():
    tracemalloc.start()
    start = time.perf_counter()
    pr = cProfile.Profile()
    pr.enable()
    run_main_once()
    pr.disable()
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats(50)
    return {
        'elapsed_seconds': elapsed,
        'mem_current_bytes': current,
        'mem_peak_bytes': peak,
        'stats': s.getvalue()
    }

def main():
    os.makedirs(os.path.join(BASE_DIR, 'profiles'), exist_ok=True)
    report_path = os.path.join(BASE_DIR, 'profiles', 'ippg_perf_report.json')
    stats_path = os.path.join(BASE_DIR, 'profiles', 'ippg_perf_stats.txt')

    # 控制测试规模
    os.environ['DEVICE_LIMIT'] = os.environ.get('DEVICE_LIMIT', '1')
    os.environ['BATCH_TRIGGER_SIZE'] = os.environ.get('BATCH_TRIGGER_SIZE', '30')
    os.environ['RUN_CSV_EXPORT'] = '0'

    # 基线：关闭并行与内存分析，保留较小批量
    os.environ['STREAMING_ENABLED'] = '0'
    os.environ['MONGO_BATCH_SIZE'] = '500'
    os.environ['DECOMP_CHUNK_SIZE'] = '500'
    os.environ['USE_AGG_DISTINCT'] = '0'
    os.environ['USE_INDEX_HINTS'] = '0'
    os.environ['SEGMENT_PARALLEL'] = '0'
    os.environ['ANALYZE_IN_MEMORY'] = '0'
    os.environ['WINDOW_HOURS'] = '24'
    baseline = profile_run()

    # 优化：启用流式、索引提示、并行与内存内分析
    os.environ['STREAMING_ENABLED'] = '1'
    os.environ['MONGO_BATCH_SIZE'] = '4000'
    os.environ['DECOMP_CHUNK_SIZE'] = '4000'
    os.environ['USE_AGG_DISTINCT'] = '1'
    os.environ['USE_INDEX_HINTS'] = '1'
    os.environ['SEGMENT_PARALLEL'] = '1'
    os.environ['ANALYZE_IN_MEMORY'] = '1'
    os.environ['SKIP_SLEEP_ANALYSIS'] = '1'
    os.environ['SLEEP_MAX_POINTS'] = '0'
    os.environ['SLEEP_SAMPLE_RATE'] = '0'
    os.environ['SEGMENT_PARALLEL_WORKERS'] = '12'
    os.environ['BATCH_PROCESS_WORKERS'] = '12'
    os.environ['WINDOW_HOURS'] = '3'
    optimized = profile_run()

    # 保留睡眠分析但进行采样的优化方案
    os.environ['STREAMING_ENABLED'] = '1'
    os.environ['MONGO_BATCH_SIZE'] = '2000'
    os.environ['DECOMP_CHUNK_SIZE'] = '2000'
    os.environ['USE_AGG_DISTINCT'] = '1'
    os.environ['USE_INDEX_HINTS'] = '1'
    os.environ['SEGMENT_PARALLEL'] = '1'
    os.environ['ANALYZE_IN_MEMORY'] = '1'
    os.environ['SKIP_SLEEP_ANALYSIS'] = '0'
    os.environ['SLEEP_MAX_POINTS'] = '200000'
    os.environ['SLEEP_SAMPLE_RATE'] = '0'
    os.environ['SEGMENT_PARALLEL_WORKERS'] = '12'
    os.environ['BATCH_PROCESS_WORKERS'] = '12'
    os.environ['WINDOW_HOURS'] = '3'
    optimized_sampling = profile_run()

    summary = {
        'baseline': {
            'elapsed_seconds': baseline['elapsed_seconds'],
            'mem_peak_bytes': baseline['mem_peak_bytes']
        },
        'optimized': {
            'elapsed_seconds': optimized['elapsed_seconds'],
            'mem_peak_bytes': optimized['mem_peak_bytes']
        },
        'optimized_with_sleep_sampling': {
            'elapsed_seconds': optimized_sampling['elapsed_seconds'],
            'mem_peak_bytes': optimized_sampling['mem_peak_bytes']
        },
        'improve_time_pct': (
            (baseline['elapsed_seconds'] - optimized['elapsed_seconds']) / baseline['elapsed_seconds'] * 100.0
            if baseline['elapsed_seconds'] > 0 else 0.0
        ),
        'improve_mem_pct': (
            (baseline['mem_peak_bytes'] - optimized['mem_peak_bytes']) / baseline['mem_peak_bytes'] * 100.0
            if baseline['mem_peak_bytes'] > 0 else 0.0
        ),
        'improve_time_pct_sampling': (
            (baseline['elapsed_seconds'] - optimized_sampling['elapsed_seconds']) / baseline['elapsed_seconds'] * 100.0
            if baseline['elapsed_seconds'] > 0 else 0.0
        ),
        'improve_mem_pct_sampling': (
            (baseline['mem_peak_bytes'] - optimized_sampling['mem_peak_bytes']) / baseline['mem_peak_bytes'] * 100.0
            if baseline['mem_peak_bytes'] > 0 else 0.0
        )
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write('=== Baseline cProfile ===\n')
        f.write(baseline['stats'])
        f.write('\n=== Optimized cProfile ===\n')
        f.write(optimized['stats'])
        f.write('\n=== Optimized With Sleep Sampling cProfile ===\n')
        f.write(optimized_sampling['stats'])

    print(f"iPPG 报告已写入: {report_path}")
    print(f"iPPG 详细统计写入: {stats_path}")
    print(f"耗时改善(禁睡眠): {summary['improve_time_pct']:.1f}% | 内存改善: {summary['improve_mem_pct']:.1f}%")
    print(f"耗时改善(睡眠采样): {summary['improve_time_pct_sampling']:.1f}% | 内存改善: {summary['improve_mem_pct_sampling']:.1f}%")

if __name__ == '__main__':
    main()