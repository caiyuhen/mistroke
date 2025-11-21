#!/usr/bin/env python3
import os
import sys
import json
import random
from typing import List, Dict, Any

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)

import iPPG_info as ippg
from process_vascular_analysis import _process_and_save_entry

def pick_devices(coll, start_utc, end_utc, k=10) -> List[str]:
    ids = ippg.group_device_ids_agg(coll, start_utc, end_utc)
    if not ids:
        ids = ippg.group_device_ids(coll, start_utc, end_utc)
    random.shuffle(ids)
    return ids[:k]

def ensure_analysis_dir() -> str:
    return ippg.timestamped_analysis_dir()

def analyze_one(coll, did, start_utc, end_utc, analysis_dir) -> Dict[str, Any]:
    out_path = ippg.process_device_rows_streaming(coll, did, start_utc, end_utc)
    fallback = False
    if out_path is None:
        rows = ippg.fetch_device_rows(coll, did, start_utc, end_utc)
        if not rows:
            return {'device_id': did, 'status': 'no_rows'}
        processed_entries, data_count, collect_count, wear_user_id = ippg.decompress_rows(rows)
        payload = {
            'deviceId': did,
            'wear_user_id': wear_user_id,
            'dataCount': data_count,
            'collectDataCount': collect_count,
            'processedData': processed_entries,
        }
        out_path = ippg.write_device_json(did, payload)
        fallback = True
    _process_and_save_entry(out_path, ippg.OUTPUT_DIR, analysis_dir)
    return {'device_id': did, 'json_path': out_path, 'status': 'ok', 'fallback': fallback}

def run():
    client, coll = ippg.connect_mongo(ippg.MONGO_CFG)
    start_utc, end_utc = ippg.utc_window_for_last_day()
    dids = pick_devices(coll, start_utc, end_utc, k=10)
    analysis_dir = ensure_analysis_dir()
    results = []
    for did in dids:
        try:
            r = analyze_one(coll, did, start_utc, end_utc, analysis_dir)
            results.append(r)
        except Exception as e:
            results.append({'device_id': did, 'status': 'error', 'error': str(e)})
    try:
        client.close()
    except Exception:
        pass
    ippg.run_advanced_7day(analysis_dir, ippg.RISK_RESULTS_DIR)
    csv_path = ippg.run_extract_device_ids_and_export(None)
    try:
        ippg.import_csv_to_mysql(csv_path, ippg.MYSQL_CFG)
    except Exception as e:
        pass
    report = {
        'window': {'start': start_utc.isoformat(), 'end': end_utc.isoformat()},
        'analysis_dir': analysis_dir,
        'devices': results,
        'summary': {
            'total': len(results),
            'ok': sum(1 for r in results if r.get('status') == 'ok'),
            'no_rows': sum(1 for r in results if r.get('status') == 'no_rows'),
            'errors': sum(1 for r in results if r.get('status') == 'error'),
        }
    }
    out_dir = os.path.join(BASE_DIR, 'profiles')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'test_ippg_report.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(out_path)

if __name__ == '__main__':
    run()