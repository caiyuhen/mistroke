#!/usr/bin/env python3
import os
import sys
import json
from typing import List, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)

import iPPG_info as ippg

def connect():
    client, coll = ippg.connect_mongo(ippg.MONGO_CFG)
    return client, coll

def list_indexes(coll):
    idxs = []
    for idx in coll.list_indexes():
        d = dict(idx)
        idxs.append({
            'name': d.get('name'),
            'key': list(d.get('key', {}).items()),
            'unique': bool(d.get('unique', False))
        })
    return idxs

def has_prefix(indexes: List[dict], keys: List[Tuple[str,int]]):
    for idx in indexes:
        k = idx['key']
        if len(k) >= len(keys) and k[:len(keys)] == keys:
            return True
    return False

def coll_stats(db, name):
    try:
        s = db.command({'collStats': name})
        return {
            'count': s.get('count'),
            'size': s.get('size'),
            'storageSize': s.get('storageSize'),
            'totalIndexSize': s.get('totalIndexSize'),
            'indexSizes': s.get('indexSizes', {})
        }
    except Exception:
        return {}

def estimate_new_index_size(count: int, avg_device_len: int):
    if not count:
        return 0
    return int(count * (avg_device_len + 8 + 64))

def main():
    client, coll = connect()
    db = client[ippg.MONGO_CFG['database']]
    name = ippg.MONGO_CFG['collection']

    idxs = list_indexes(coll)
    stats = coll_stats(db, name)
    count = stats.get('count') or coll.estimated_document_count()
    sample = coll.find_one({}, {'deviceId': 1}) or {}
    dev = sample.get('deviceId')
    avg_len = len(str(dev)) if dev is not None else 16

    q1_needed = [('deviceId', 1), ('collectTime', 1)]
    q2_needed = [('collectTime', 1)]
    q3_needed_sort_ct_dev = [('collectTime', 1), ('deviceId', 1)]
    q3_needed_sort_dev_ct = [('deviceId', 1), ('collectTime', 1)]

    q1_match = has_prefix(idxs, q1_needed)
    q2_match = has_prefix(idxs, q2_needed)
    q3_match_ct_dev = has_prefix(idxs, q3_needed_sort_ct_dev)
    q3_match_dev_ct = has_prefix(idxs, q3_needed_sort_dev_ct)

    suggestions = []
    if not q1_match:
        suggestions.append({'create': q1_needed, 'reason': 'distinct deviceId with collectTime range'})
    if not q2_match:
        suggestions.append({'create': q2_needed, 'reason': 'time window scans'})
    if not (q3_match_ct_dev or q3_match_dev_ct):
        suggestions.append({'create': q3_needed_sort_ct_dev, 'reason': 'streaming sort by collectTime,deviceId; consider changing sort order to match'})

    est_sizes = []
    for s in suggestions:
        est_sizes.append({'keys': s['create'], 'estimatedSizeBytes': estimate_new_index_size(count, avg_len)})

    report = {
        'cluster': {'host': ippg.MONGO_CFG['host'], 'db': ippg.MONGO_CFG['database'], 'collection': name},
        'existingIndexes': idxs,
        'collectionStats': stats,
        'queryPatterns': [
            {'name': 'distinct_device_in_time', 'filter': ['collectTime: $gte,$lte'], 'distinct': 'deviceId', 'neededIndex': q1_needed, 'matched': q1_match},
            {'name': 'find_device_time_sorted', 'filter': ['deviceId', 'collectTime: $gte,$lte'], 'sort': ['collectTime:1'], 'neededIndex': q1_needed, 'matched': q1_match},
            {'name': 'stream_all_by_time', 'filter': ['collectTime: $gte,$lte'], 'sort': ['collectTime:1','deviceId:1'], 'neededIndex': q3_needed_sort_ct_dev, 'matched': q3_match_ct_dev},
        ],
        'suggestions': suggestions,
        'estimatedNewIndexSizes': est_sizes
    }

    out_dir = os.path.join(BASE_DIR, 'profiles')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'mongo_index_report.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(out_path)

    try:
        client.close()
    except Exception:
        pass

if __name__ == '__main__':
    main()