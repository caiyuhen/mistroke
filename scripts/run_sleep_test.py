import os
import sys
import numpy as np
from datetime import datetime, timedelta
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ppg_sleep_analysis import PPGSleepAnalyzer

def main():
    N = 2000
    rng = np.random.default_rng(0)
    ppg = rng.normal(size=N).astype(float)
    base = datetime(2024, 8, 1, 12, 0, 0)
    ts = np.array([np.datetime64(base + timedelta(seconds=i*8), 's') for i in range(N)])
    a = PPGSleepAnalyzer(125)
    for name, fn in (
        ('apnea', a.analyze_sleep_apnea),
        ('spo2', a.analyze_nocturnal_spo2),
        ('bp', a.analyze_blood_pressure_rhythm),
    ):
        try:
            r = fn(ppg, ts)
            print(name, 'ok keys', list(r.keys()))
        except Exception as e:
            import traceback; print(name, 'error'); traceback.print_exc()

if __name__ == '__main__':
    main()