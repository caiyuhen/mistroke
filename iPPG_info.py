import os
import sys
import json
import shutil
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import time

import pymongo
from pymongo import MongoClient
from pymongo.read_preferences import SecondaryPreferred, Primary

try:
    import orjson as _orjson
except Exception:
    _orjson = None

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pymysql

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
ANALYSIS_DIR_BASE = os.path.join(BASE_DIR, "analysis_results")
RISK_RESULTS_DIR = os.path.join(BASE_DIR, "risk_results")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "iPPG_info.log")

MONGO_CFG = {
    "host": "mongoreplica29ed1d62f12a1.mongodb.cn-beijing.volces.com",
    "port": 3717,
    "username": "apps-wr",
    "password": "eEcc7U!nNM3ivzC^f",
    "database": "andun_1",
    "collection": "device_collect_compress_data",
}

MYSQL_CFG = {
    "host": "rm-2ze3zk2327k92186do.mysql.rds.aliyuncs.com", #101.200.161.50
    "port": 3306,
    "user": "infarction_user",
    "password": "Bm*PiyeQjD6cGii4",
    "database": "andun_health",
    "table": "health_analysis",
}

def get_mongo_cfg() -> Dict[str, Any]:
    return {
        "host": os.environ.get("MONGO_HOST", MONGO_CFG["host"]),
        "port": int(os.environ.get("MONGO_PORT", str(MONGO_CFG["port"]))),
        "username": os.environ.get("MONGO_USERNAME", MONGO_CFG["username"]),
        "password": os.environ.get("MONGO_PASSWORD", MONGO_CFG["password"]),
        "database": os.environ.get("MONGO_DATABASE", MONGO_CFG["database"]),
        "collection": os.environ.get("MONGO_COLLECTION", MONGO_CFG["collection"]),
    }

def get_mysql_cfg() -> Dict[str, Any]:
    return {
        "host": os.environ.get("MYSQL_HOST", MYSQL_CFG["host"]),
        "port": int(os.environ.get("MYSQL_PORT", str(MYSQL_CFG["port"]))),
        "user": os.environ.get("MYSQL_USER", MYSQL_CFG["user"]),
        "password": os.environ.get("MYSQL_PASSWORD", MYSQL_CFG["password"]),
        "database": os.environ.get("MYSQL_DATABASE", MYSQL_CFG["database"]),
        "table": os.environ.get("MYSQL_TABLE", MYSQL_CFG["table"]),
    }

class TTLCache:
    def __init__(self, ttl: float = 300.0, maxsize: int = 64) -> None:
        self.ttl = ttl
        self.maxsize = maxsize
        self.store: Dict[str, Tuple[Any, float]] = {}
    def get(self, key: str) -> Optional[Any]:
        v = self.store.get(key)
        if not v:
            return None
        val, ts = v
        if (time.time() - ts) > self.ttl:
            try:
                del self.store[key]
            except Exception:
                pass
            return None
        return val
    def set(self, key: str, val: Any) -> None:
        if len(self.store) >= self.maxsize:
            try:
                oldest_key = min(self.store.items(), key=lambda x: x[1][1])[0] if self.store else None
                if oldest_key:
                    del self.store[oldest_key]
            except Exception:
                self.store.clear()
        self.store[key] = (val, time.time())

DEVICE_IDS_CACHE = TTLCache(ttl=600.0, maxsize=128)

def setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            root.removeHandler(h)
        except Exception:
            pass
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    try:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        pass
    root.propagate = False
    logging.raiseExceptions = False

def _paging_state_path() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, "paging_state.json")

def _load_paging_state() -> Dict[str, Any]:
    try:
        with open(_paging_state_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_paging_state(state: Dict[str, Any]) -> None:
    try:
        with open(_paging_state_path(), "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
    except Exception:
        pass

def _get_last_ct_state(device_id: str) -> Optional[datetime]:
    try:
        st = _load_paging_state()
        v = st.get(device_id, {}).get("last_ct")
        if not v:
            return None
        dt = to_datetime(v)
        return dt
    except Exception:
        return None

def _set_last_ct_state(device_id: str, dt: Optional[datetime]) -> None:
    try:
        if dt is None:
            return
        st = _load_paging_state()
        st[device_id] = {"last_ct": dt.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat()}
        _save_paging_state(st)
    except Exception:
        pass

def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RISK_RESULTS_DIR, exist_ok=True)

def clear_output_dir() -> None:
    if os.path.isdir(OUTPUT_DIR):
        with os.scandir(OUTPUT_DIR) as it:
            for entry in it:
                try:
                    if entry.is_file():
                        os.remove(entry.path)
                except Exception:
                    pass

def timestamped_analysis_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_dir = f"{ANALYSIS_DIR_BASE}_{ts}"
    if os.path.isdir(ANALYSIS_DIR_BASE):
        try:
            os.rename(ANALYSIS_DIR_BASE, final_dir)
        except Exception:
            alt = ANALYSIS_DIR_BASE + "_renamed_" + ts
            try:
                os.rename(ANALYSIS_DIR_BASE, alt)
            except Exception:
                pass
    os.makedirs(final_dir, exist_ok=True)
    return final_dir

def connect_mongo(cfg: Dict[str, Any]) -> Tuple[MongoClient, pymongo.collection.Collection]:
    uri = (
        f"mongodb://{cfg['username']}:{cfg['password']}@"
        f"{cfg['host']}:{cfg['port']}/{cfg['database']}?authSource=admin"
    )
    rp_env = os.environ.get("READ_PREFERENCE", "primary")
    rp = SecondaryPreferred() if rp_env == "secondaryPreferred" else Primary()
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
        socketTimeoutMS=120000,
        compressors=["zstd", "snappy"],
        maxPoolSize=max(64, (os.cpu_count() or 4) * 8),
        minPoolSize=max(16, (os.cpu_count() or 4)),
        retryReads=True,
        read_preference=rp,
    )
    _ = client.admin.command("ping")
    db = client[cfg["database"]]
    coll = db[cfg["collection"]]
    return client, coll

def ensure_mongo_indexes(coll: pymongo.collection.Collection) -> None:
    try:
        coll.create_index([
            ("deviceId", pymongo.ASCENDING),
            ("collectTime", pymongo.DESCENDING),
        ], name="deviceId_1_collectTime_-1")
        coll.create_index([
            ("collectTime", pymongo.ASCENDING),
            ("deviceId", pymongo.ASCENDING),
        ], name="collectTime_1_deviceId_1")
    except Exception:
        pass

def query_projection() -> Dict[str, int]:
    if os.environ.get("THIN_PROJECTION", "0") == "1":
        return {"deviceId": 1, "collectTime": 1, "collectData": 1}
    return {"deviceId": 1, "collectTime": 1, "createTime": 1, "collectData": 1, "wear_user_id": 1, "wearUserId": 1}

def time_window_from_env() -> Tuple[datetime, datetime]:
    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=1)
    return start_utc, end_utc

def _index_pref() -> Tuple[str, int]:
    val = os.environ.get("PREFERRED_INDEX", "ct_dev_asc")
    if val == "ct_dev_asc":
        return "collectTime_1_deviceId_1", 1
    return "deviceId_1_collectTime_-1", -1

def _json_dump(data: Dict[str, Any]) -> bytes:
    if _orjson is not None:
        try:
            return _orjson.dumps(data)
        except Exception:
            pass
    return json.dumps(data, ensure_ascii=False).encode("utf-8")

def write_device_json(device_id: str, payload: Dict[str, Any]) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = f"{device_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    fpath = os.path.join(OUTPUT_DIR, fname)
    try:
        if os.environ.get("DRY_RUN", "0") == "1":
            return fpath
        with open(fpath, "wb") as f:
            f.write(_json_dump(payload))
        logging.info(f"写出: {fpath}")
    except Exception as e:
        logging.warning(f"写出失败: {fpath} {e}")
    return fpath

def to_datetime(val: Any) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val))
    except Exception:
        return None

def extract_bytes(data: Any) -> Optional[bytes]:
    if data is None:
        return None
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    try:
        return bytes(data)
    except Exception:
        return None

from decompressed_comp_ppg_data import (
    get_compression_byte_data,
    check_bytes_compress_data,
    check_compression_data,
    uncompression_and_check_data,
    uncompress_data_upsample,
)

def decompress_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int, Optional[str]]:
    res_list: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []
    for r in rows:
        cbytes = extract_bytes(r.get("collectData"))
        ctime = to_datetime(r.get("collectTime"))
        if cbytes and ctime:
            res_list.append({"collectTime": ctime, "collectData": cbytes})
            valid_rows.append(r)
    collect_count = len(res_list)
    wear_user_id = None
    if rows:
        first = rows[0]
        wear_user_id = first.get("wear_user_id") or first.get("wearUserId")
    if collect_count == 0:
        return [], 0, 0, wear_user_id
    compression_data_bytes_list = get_compression_byte_data(res_list)
    compression_data_int_list = check_bytes_compress_data(compression_data_bytes_list)
    idx_map: List[int] = []
    filtered_int_list: List[List[int]] = []
    filtered_rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(compression_data_int_list):
        if item is not None:
            idx_map.append(idx)
            filtered_int_list.append(item)
            filtered_rows.append(valid_rows[idx])
    if not filtered_int_list:
        return [], 0, collect_count, wear_user_id
    _ = check_compression_data(filtered_int_list)
    origin_ppg_data_list, hz_list = uncompression_and_check_data(filtered_int_list, request_tag="")
    upsampled_data_list = uncompress_data_upsample(origin_ppg_data_list, target_fs=250, rates=hz_list)
    processed_entries: List[Dict[str, Any]] = []
    for i, (orig, upsampled, row) in enumerate(zip(origin_ppg_data_list, upsampled_data_list, filtered_rows)):
        ctime = to_datetime(row.get("collectTime"))
        crtime = to_datetime(row.get("createTime"))
        processed_entries.append({
            "index": i,
            "collectTime": ctime.strftime("%Y-%m-%d %H:%M:%S") if ctime else "",
            "createTime": crtime.strftime("%Y-%m-%d %H:%M:%S.%f") if crtime else "",
            "collectTime_beijing": (ctime.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S") if ctime and ctime.tzinfo else ""),
            "createTime_beijing": (crtime.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S") if crtime and crtime.tzinfo else ""),
            "originalDataSize": len(orig) if isinstance(orig, list) else 0,
            "decompressedData": upsampled if isinstance(upsampled, list) else [],
        })
    return processed_entries, len(processed_entries), collect_count, wear_user_id

def _decompress_rows_chunk(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
    res_list: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []
    for r in rows:
        cbytes = extract_bytes(r.get("collectData"))
        ctime = r.get("collectTime")
        dt = ctime if isinstance(ctime, datetime) else to_datetime(ctime)
        if cbytes and dt:
            res_list.append({"collectTime": dt, "collectData": cbytes})
            valid_rows.append(r)
    collect_count = len(res_list)
    if collect_count == 0:
        return [], 0, 0
    compression_data_bytes_list = get_compression_byte_data(res_list)
    compression_data_int_list = check_bytes_compress_data(compression_data_bytes_list)
    filtered_int_list: List[List[int]] = []
    filtered_rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(compression_data_int_list):
        if item is not None:
            filtered_int_list.append(item)
            filtered_rows.append(valid_rows[idx])
    if not filtered_int_list:
        return [], 0, collect_count
    _ = check_compression_data(filtered_int_list)
    origin_ppg_data_list, hz_list = uncompression_and_check_data(filtered_int_list, request_tag="")
    upsampled_data_list = uncompress_data_upsample(origin_ppg_data_list, target_fs=250, rates=hz_list)
    processed_entries: List[Dict[str, Any]] = []
    for i, (orig, upsampled, row) in enumerate(zip(origin_ppg_data_list, upsampled_data_list, filtered_rows)):
        ctime = to_datetime(row.get("collectTime"))
        crtime = to_datetime(row.get("createTime"))
        processed_entries.append({
            "index": i,
            "collectTime": ctime.strftime("%Y-%m-%d %H:%M:%S") if ctime else "",
            "createTime": crtime.strftime("%Y-%m-%d %H:%M:%S.%f") if crtime else "",
            "collectTime_beijing": (ctime.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S") if ctime and ctime.tzinfo else ""),
            "createTime_beijing": (crtime.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S") if crtime and crtime.tzinfo else ""),
            "originalDataSize": len(orig) if isinstance(orig, list) else 0,
            "decompressedData": upsampled if isinstance(upsampled, list) else [],
        })
    return processed_entries, len(processed_entries), collect_count

def process_device_rows_streaming(coll, device_id: str, start_utc: datetime, end_utc: datetime) -> Optional[str]:
    projection = query_projection()
    batch_sz = int(os.environ.get("MONGO_BATCH_SIZE", "4000"))
    chunk_sz = int(os.environ.get("DECOMP_CHUNK_SIZE", "4000"))
    try:
        idx, sdir = _index_pref()
        paging = os.environ.get("PAGING_ENABLED", "0") == "1"
        mt = int(os.environ.get("QUERY_MAX_TIME_MS", "60000"))
        if not paging:
            cursor = coll.find({"deviceId": device_id, "collectTime": {"$gte": start_utc, "$lte": end_utc}}, projection=projection, no_cursor_timeout=True, max_time_ms=(mt if mt > 0 else None)).hint(idx).sort("collectTime", sdir).batch_size(batch_sz)
        else:
            cursor = None
    except Exception:
        return None
    processed_entries: List[Dict[str, Any]] = []
    total_collect = 0
    wear_user_id = None
    buf: List[Dict[str, Any]] = []
    chunks: List[List[Dict[str, Any]]] = []
    try:
        if os.environ.get("PAGING_ENABLED", "0") == "1":
            page_sz = int(os.environ.get("PAGE_SIZE", str(batch_sz)))
            resume_enabled = os.environ.get("PAGING_RESUME", "1") == "1"
            last_ct: Optional[datetime] = None
            if resume_enabled:
                prev = _get_last_ct_state(device_id)
                if prev is not None and prev >= start_utc and prev <= end_utc:
                    last_ct = prev
                    logging.info(f"断点续读: 从持久化 last_ct={last_ct.isoformat()} 继续")
            while True:
                q = {"deviceId": device_id, "collectTime": {"$gte": start_utc, "$lte": end_utc}}
                if last_ct is not None:
                    if sdir == 1:
                        q["collectTime"]["$gt"] = last_ct
                    else:
                        q["collectTime"]["$lt"] = last_ct
                t0 = time.perf_counter()
                page = list(coll.find(q, projection=projection, max_time_ms=(mt if mt > 0 else None)).hint(idx).sort("collectTime", sdir).limit(page_sz))
                logging.info(f"分页读取: deviceId={device_id}, size={len(page)}, 耗时ms={round((time.perf_counter()-t0)*1000)}")
                if not page:
                    break
                if wear_user_id is None and page:
                    wear_user_id = page[0].get("wear_user_id") or page[0].get("wearUserId")
                last_ct = page[-1].get("collectTime") if page else last_ct
                _set_last_ct_state(device_id, last_ct)
                chunks.append(page)
        else:
            for doc in cursor:
                if wear_user_id is None:
                    wear_user_id = doc.get("wear_user_id") or doc.get("wearUserId")
                buf.append(doc)
                if len(buf) >= chunk_sz:
                    chunks.append(buf)
                    buf = []
            if buf:
                chunks.append(buf)
    finally:
        try:
            if cursor is not None:
                cursor.close()
        except Exception:
            pass
    if chunks:
        etype = os.environ.get("DECOMP_EXECUTOR", "thread")
        env_workers = os.environ.get("DECOMP_WORKERS", "12")
        max_workers = max(1, (os.cpu_count() or 2) - 1) if not (env_workers and env_workers.isdigit()) else max(1, int(env_workers))
        if etype == "process":
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_decompress_rows_chunk, c) for c in chunks]
                for fut in as_completed(futs):
                    try:
                        entries, data_cnt, collect_cnt = fut.result()
                        processed_entries.extend(entries)
                        total_collect += collect_cnt
                    except Exception:
                        pass
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_decompress_rows_chunk, c) for c in chunks]
                for fut in as_completed(futs):
                    try:
                        entries, data_cnt, collect_cnt = fut.result()
                        processed_entries.extend(entries)
                        total_collect += collect_cnt
                    except Exception:
                        pass
    payload = {
        "deviceId": device_id,
        "wear_user_id": wear_user_id,
        "dataCount": len(processed_entries),
        "collectDataCount": total_collect,
        "processedData": processed_entries,
    }
    return write_device_json(device_id, payload)

def process_batch(json_paths: List[str], input_dir: str, output_dir: str) -> None:
    from process_vascular_analysis import _process_and_save_entry
    exec_type = os.environ.get("BATCH_PROCESS_EXECUTOR", "process")
    default_workers = max(1, min(12, (os.cpu_count() or 2)))
    env_workers = os.environ.get("BATCH_PROCESS_WORKERS", "12")
    max_workers = default_workers if not (env_workers and env_workers.isdigit()) else max(1, int(env_workers))
    if exec_type == "process":
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_process_and_save_entry, p, input_dir, output_dir) for p in json_paths]
            for fut in as_completed(futs):
                try:
                    _ = fut.result()
                except Exception:
                    pass
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_process_and_save_entry, p, input_dir, output_dir) for p in json_paths]
            for fut in as_completed(futs):
                try:
                    _ = fut.result()
                except Exception:
                    pass

def run_advanced_7day(input_dir: str, output_dir: str) -> None:
    import importlib.util
    mod_path = os.path.join(BASE_DIR, '7day_advanced_analytics.py')
    spec = importlib.util.spec_from_file_location('advanced7', mod_path)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assessor = getattr(mod, 'Advanced7DayRiskAssessment')()
        _ = assessor.batch_process_files(input_dir, output_dir)

def run_extract_device_ids_and_export(csv_out: Optional[str] = None) -> str:
    from scripts.extract_device_ids import main as extract_main
    try:
        sys.argv = [sys.argv[0]] + ([] if csv_out is None else ["--csv-path", csv_out])
        extract_main()
    except SystemExit:
        pass
    target = csv_out or os.path.join(BASE_DIR, "device_wear_health_export.csv")
    return target

def import_csv_to_mysql(csv_path: str, cfg: Dict[str, Any]) -> None:
    conn = None
    try:
        conn = pymysql.connect(host=cfg["host"], port=cfg["port"], user=cfg["user"], password=cfg["password"], database=cfg["database"], charset="utf8mb4", local_infile=True)
        table = cfg["table"]
        cols = (
            "device_id, wear_user_id, source_date, @create_time, heart_status, spine_status, lymph_status, @pericardium_status, small_intestine_status, lung_status, stomach_status, sys_mi_risk_score, @sys_stroke_risk_score, mi_risk_level2, stroke_risk_level2"
        )
        sql = (
            f"LOAD DATA LOCAL INFILE %s INTO TABLE `{table}` CHARACTER SET utf8mb4 "
            f"FIELDS TERMINATED BY ',' ENCLOSED BY '""' ESCAPED BY '' "
            f"LINES TERMINATED BY '\r\n' IGNORE 1 LINES ({cols}) "
            f"SET create_time=@create_time, sys_stroker_risk_score=@sys_stroke_risk_score, pericardium_status=@pericardium_status"
        )
        with conn.cursor() as cur:
            cur.execute(sql, (csv_path,))
        conn.commit()
        logging.info(f"导入CSV到MySQL成功: {csv_path}")
    except Exception as e:
        logging.warning(f"LOAD DATA 导入失败，回退逐行插入: {e}")
        try:
            import csv
            if conn is None:
                conn = pymysql.connect(host=cfg["host"], port=cfg["port"], user=cfg["user"], password=cfg["password"], database=cfg["database"], charset="utf8mb4")
            db_cols = [
                "device_id",
                "wear_user_id",
                "source_date",
                "create_time",
                "heart_status",
                "spine_status",
                "lymph_status",
                "pericardium_status",
                "small_intestine_status",
                "lung_status",
                "stomach_status",
                "sys_mi_risk_score",
                "sys_stroker_risk_score",
                "mi_risk_level2",
                "stroke_risk_level2",
            ]
            placeholders = ",".join(["%s"] * len(db_cols))
            insert_sql = f"INSERT INTO `{cfg['table']}` ({','.join(db_cols)}) VALUES ({placeholders})"
            batch_sz = int(os.environ.get("CSV_INSERT_BATCH", "1000"))
            total = 0
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                batch_vals: List[List[Any]] = []
                with conn.cursor() as cur:
                    for r in reader:
                        vals = [
                            r.get("device_id"),
                            r.get("wear_user_id"),
                            r.get("source_date"),
                            r.get("create_time"),
                            r.get("heart_status"),
                            r.get("spine_status"),
                            r.get("lymph_status"),
                            r.get("pericardium_status"),
                            r.get("small_intestine_status"),
                            r.get("lung_status"),
                            r.get("stomach_status"),
                            r.get("sys_mi_risk_score"),
                            r.get("sys_stroke_risk_score"),
                            r.get("mi_risk_level2"),
                            r.get("stroke_risk_level2"),
                        ]
                        batch_vals.append(vals)
                        if len(batch_vals) >= batch_sz:
                            cur.executemany(insert_sql, batch_vals)
                            total += len(batch_vals)
                            batch_vals.clear()
                    if batch_vals:
                        cur.executemany(insert_sql, batch_vals)
                        total += len(batch_vals)
            conn.commit()
            logging.info(f"逐行插入导入成功: {total} 行")
        except Exception as e2:
            logging.error(f"导入CSV到MySQL失败: {e2}")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass

def _process_one_device(coll, did: str, start_utc: datetime, end_utc: datetime, analysis_dir: str) -> Optional[str]:
    t0 = time.perf_counter()
    if os.environ.get("STREAMING_ENABLED", "1") == "1":
        out_path = process_device_rows_streaming(coll, did, start_utc, end_utc)
        if out_path is None:
            rows = fetch_device_rows(coll, did, start_utc, end_utc)
            if not rows:
                return None
            processed_entries, data_count, collect_count, wear_user_id = decompress_rows(rows)
            payload = {
                "deviceId": did,
                "wear_user_id": wear_user_id,
                "dataCount": data_count,
                "collectDataCount": collect_count,
                "processedData": processed_entries,
            }
            out_path = write_device_json(did, payload)
        logging.info(f"设备读取耗时ms={round((time.perf_counter()-t0)*1000)} deviceId={did}")
        return out_path
    rows = fetch_device_rows(coll, did, start_utc, end_utc)
    if not rows:
        return None
    processed_entries, data_count, collect_count, wear_user_id = decompress_rows(rows)
    payload = {
        "deviceId": did,
        "wear_user_id": wear_user_id,
        "dataCount": data_count,
        "collectDataCount": collect_count,
        "processedData": processed_entries,
    }
    out = write_device_json(did, payload)
    logging.info(f"设备读取耗时ms={round((time.perf_counter()-t0)*1000)} deviceId={did}")
    return out

def process_devices_parallel(coll, device_ids: List[str], start_utc: datetime, end_utc: datetime, analysis_dir: str) -> List[str]:
    default_workers = max(1, min(12, (os.cpu_count() or 2)))
    env_workers = os.environ.get("DEVICE_PARALLEL_WORKERS", "12")
    max_workers = default_workers if not (env_workers and env_workers.isdigit()) else max(1, int(env_workers))
    batch: List[str] = []
    trigger_n = int(os.environ.get("BATCH_TRIGGER_SIZE", "30"))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_process_one_device, coll, did, start_utc, end_utc, analysis_dir): did for did in device_ids}
        for fut in as_completed(futs):
            try:
                out_path = fut.result()
                if out_path:
                    batch.append(out_path)
            except Exception:
                pass
    return batch

def group_device_ids(coll, start_utc: datetime, end_utc: datetime) -> List[str]:
    try:
        k = f"distinct:{start_utc.isoformat()}:{end_utc.isoformat()}"
        cv = DEVICE_IDS_CACHE.get(k)
        if cv is not None:
            return cv
        t0 = time.perf_counter()
        mt = int(os.environ.get("QUERY_MAX_TIME_MS", "60000"))
        res = [str(d) for d in coll.distinct("deviceId", {"collectTime": {"$gte": start_utc, "$lte": end_utc}}, maxTimeMS=(mt if mt > 0 else None)) if d]
        logging.info(f"distinct 耗时ms={round((time.perf_counter()-t0)*1000)} 返回={len(res)}")
        DEVICE_IDS_CACHE.set(k, res)
        return res
    except Exception:
        return []

def group_device_ids_agg(coll, start_utc: datetime, end_utc: datetime) -> List[str]:
    try:
        k = f"agg:{start_utc.isoformat()}:{end_utc.isoformat()}"
        cv = DEVICE_IDS_CACHE.get(k)
        if cv is not None:
            return cv
        pipeline = [
            {"$match": {"collectTime": {"$gte": start_utc, "$lte": end_utc}}},
            {"$group": {"_id": "$deviceId"}},
        ]
        t0 = time.perf_counter()
        idx, _ = _index_pref()
        mt = int(os.environ.get("QUERY_MAX_TIME_MS", "60000"))
        result = coll.aggregate(pipeline, allowDiskUse=True, hint=idx, maxTimeMS=(mt if mt > 0 else None))
        result = result.batch_size(int(os.environ.get("AGG_BATCH_SIZE", "10000")))
        out = [str(doc.get("_id")) for doc in result if doc.get("_id")]
        logging.info(f"聚合 耗时ms={round((time.perf_counter()-t0)*1000)} 返回={len(out)}")
        DEVICE_IDS_CACHE.set(k, out)
        return out
    except Exception:
        return []

def fetch_device_rows(coll, device_id: str, start_utc: datetime, end_utc: datetime) -> List[Dict[str, Any]]:
    projection = query_projection()
    try:
        bs = int(os.environ.get("MONGO_BATCH_SIZE", "4000"))
        use_hint = os.environ.get("USE_INDEX_HINTS", "1") == "1"
        idx, sdir = _index_pref()
        mt = int(os.environ.get("QUERY_MAX_TIME_MS", "60000"))
        if use_hint:
            logging.info(
                "Mongo 查询(find): "
                + f"deviceId={device_id}, collectTime >= {start_utc.isoformat()} 且 <= {end_utc.isoformat()}, "
                + f"projection=['deviceId','collectTime','createTime','collectData','wear_user_id','wearUserId'], sort=collectTime {'DESC' if sdir==-1 else 'ASC'}, batch_size={bs}, hint={idx}"
            )
            t0 = time.perf_counter()
            cursor = coll.find({"deviceId": device_id, "collectTime": {"$gte": start_utc, "$lte": end_utc}}, projection=projection, no_cursor_timeout=True, max_time_ms=(mt if mt > 0 else None)).hint(idx).sort("collectTime", sdir).batch_size(bs)
            docs = list(cursor)
            logging.info(f"find 耗时ms={round((time.perf_counter()-t0)*1000)} 返回={len(docs)}")
            return docs
        else:
            logging.info(
                "Mongo 查询(find): "
                + f"deviceId={device_id}, collectTime >= {start_utc.isoformat()} 且 <= {end_utc.isoformat()}, "
                + f"projection=['deviceId','collectTime','createTime','collectData','wear_user_id','wearUserId'], sort=collectTime {'DESC' if sdir==-1 else 'ASC'}, batch_size={bs}"
            )
            t0 = time.perf_counter()
            cursor = coll.find({"deviceId": device_id, "collectTime": {"$gte": start_utc, "$lte": end_utc}}, projection=projection, no_cursor_timeout=True, max_time_ms=(mt if mt > 0 else None)).sort("collectTime", sdir).batch_size(bs)
            docs = list(cursor)
            logging.info(f"find 耗时ms={round((time.perf_counter()-t0)*1000)} 返回={len(docs)}")
            return docs
    except Exception:
        return []

def main() -> None:
    setup_logging()
    logging.info("启动 iPPG 流水线")
    os.environ.setdefault("BATCH_PROCESS_EXECUTOR", "process")
    os.environ.setdefault("BATCH_PROCESS_WORKERS", "12")
    os.environ.setdefault("SEGMENT_PARALLEL", "1")
    os.environ.setdefault("SEGMENT_EXECUTOR", "thread")
    os.environ.setdefault("SEGMENT_PARALLEL_WORKERS", "8")
    os.environ.setdefault("SKIP_ANALYSIS", "0")
    os.environ.setdefault("RUN_CSV_EXPORT", "1")
    ensure_dirs()
    clear_output_dir()
    analysis_dir = timestamped_analysis_dir()
    try:
        client, coll = connect_mongo(get_mongo_cfg())
        try:
            ensure_mongo_indexes(coll)
        except Exception:
            pass
    except Exception as e:
        logging.error(f"Mongo连接失败: {e}")
        return
    start_utc, end_utc = time_window_from_env()
    logging.info(f"检索窗口(UTC): {start_utc.isoformat()} ~ {end_utc.isoformat()}")
    read_t0 = time.perf_counter()
    if os.environ.get("USE_AGG_DISTINCT", "1") == "1":
        device_ids = group_device_ids_agg(coll, start_utc, end_utc)
    else:
        device_ids = group_device_ids(coll, start_utc, end_utc)
    if not device_ids and os.environ.get("STREAM_ALL", "1") != "1":
        logging.info("未检索到设备")
    device_ids = sorted([str(x) for x in device_ids])
    try:
        _auto_tune_if_needed(coll, start_utc, end_utc, analysis_dir, device_ids)
    except Exception:
        pass
    try:
        if os.environ.get("PAGING_ENABLED") is None:
            total_ids = len(device_ids)
            thr = int(os.environ.get("AUTO_PAGING_THRESHOLD", "20000"))
            if total_ids >= thr:
                os.environ["PAGING_ENABLED"] = "1"
                if os.environ.get("PAGE_SIZE") is None:
                    os.environ["PAGE_SIZE"] = os.environ.get("MONGO_BATCH_SIZE", "4000")
                logging.info(f"启用分页读取: 设备数={total_ids}, PAGE_SIZE={os.environ.get('PAGE_SIZE')}")
    except Exception:
        pass
    batch: List[str] = []
    limit_env = os.environ.get("DEVICE_LIMIT")
    if os.environ.get("STREAM_ALL", "1") == "0":
        projection = {"deviceId": 1, "collectTime": 1, "createTime": 1, "collectData": 1, "wear_user_id": 1, "wearUserId": 1}
        bs = int(os.environ.get("MONGO_BATCH_SIZE", "4000"))
        use_hint = os.environ.get("USE_INDEX_HINTS", "1") == "1"
        idx, sdir = _index_pref()
        if use_hint:
            t0 = time.perf_counter()
            cursor = coll.find({"collectTime": {"$gte": start_utc, "$lte": end_utc}}, projection=projection, no_cursor_timeout=True).hint(idx).sort("collectTime", sdir).batch_size(bs)
            logging.info(f"全量游标创建耗时ms={round((time.perf_counter()-t0)*1000)}")
        else:
            t0 = time.perf_counter()
            cursor = coll.find({"collectTime": {"$gte": start_utc, "$lte": end_utc}}, projection=projection, no_cursor_timeout=True).sort("collectTime", sdir).batch_size(bs)
            logging.info(f"全量游标创建耗时ms={round((time.perf_counter()-t0)*1000)}")
        current_id = None
        wear_user_id = None
        buf: List[Dict[str, Any]] = []
        processed_entries: List[Dict[str, Any]] = []
        total_collect = 0
        chunk_sz = int(os.environ.get("DECOMP_CHUNK_SIZE", "4000"))
        processed_devices = 0
        stopped = False
        try:
            for doc in cursor:
                did = str(doc.get("deviceId"))
                if current_id is None:
                    current_id = did
                    wear_user_id = doc.get("wear_user_id") or doc.get("wearUserId")
                if did != current_id:
                    payload = {
                        "deviceId": current_id,
                        "wear_user_id": wear_user_id,
                        "dataCount": len(processed_entries),
                        "collectDataCount": total_collect,
                        "processedData": processed_entries,
                    }
                    out_path = write_device_json(current_id, payload)
                    batch.append(out_path)
                    processed_devices += 1
                    if limit_env and limit_env.isdigit() and processed_devices >= int(limit_env):
                        stopped = True
                        break
                    current_id = did
                    wear_user_id = doc.get("wear_user_id") or doc.get("wearUserId")
                    buf.clear()
                    processed_entries.clear()
                    total_collect = 0
                buf.append(doc)
                if len(buf) >= chunk_sz:
                    entries, data_cnt, collect_cnt = _decompress_rows_chunk(buf)
                    processed_entries.extend(entries)
                    total_collect += collect_cnt
                    buf.clear()
            if current_id is not None and not stopped:
                if buf:
                    entries, data_cnt, collect_cnt = _decompress_rows_chunk(buf)
                    processed_entries.extend(entries)
                    total_collect += collect_cnt
                payload = {
                    "deviceId": current_id,
                    "wear_user_id": wear_user_id,
                    "dataCount": len(processed_entries),
                    "collectDataCount": total_collect,
                    "processedData": processed_entries,
                }
                out_path = write_device_json(current_id, payload)
                batch.append(out_path)
        finally:
            try:
                cursor.close()
            except Exception:
                pass
    else:
        if limit_env and limit_env.isdigit():
            device_ids = device_ids[:int(limit_env)]
        batch = process_devices_parallel(coll, device_ids, start_utc, end_utc, analysis_dir)
    if batch:
        if os.environ.get("SKIP_ANALYSIS", "0") != "1":
            logging.info("处理批次分析")
            process_batch(batch, OUTPUT_DIR, analysis_dir)
    logging.info(f"读取阶段耗时ms={round((time.perf_counter()-read_t0)*1000)}")
    try:
        client.close()
    except Exception:
        pass
    if os.environ.get("SKIP_ANALYSIS", "0") != "1":
        logging.info("开始7天高级风险评估")
        run_advanced_7day(analysis_dir, RISK_RESULTS_DIR)
    if os.environ.get("RUN_CSV_EXPORT", "1") == "1":
        logging.info("执行设备ID提取与CSV导出")
        csv_out = None
        try:
            csv_out = run_extract_device_ids_and_export()
            logging.info(f"CSV生成路径: {csv_out}")
        except Exception as e:
            logging.error(f"CSV生成失败: {e}")
        try:
            if csv_out and os.path.isfile(csv_out):
                import_csv_to_mysql(csv_out, get_mysql_cfg())
                logging.info("CSV导入MySQL完成")
            else:
                logging.warning("CSV文件不存在，跳过导入MySQL")
        except Exception as e:
            logging.error(f"CSV导入MySQL失败: {e}")

if __name__ == "__main__":
    main()
def _apply_settings(settings: Dict[str, Any]) -> None:
    for k, v in settings.items():
        os.environ[str(k)] = str(v)

def _measure_read_time(coll, sample_ids: List[str], start_utc: datetime, end_utc: datetime, analysis_dir: str, settings: Dict[str, Any]) -> float:
    old_env = os.environ.copy()
    try:
        tmp = settings.copy()
        tmp.update({"READ_ONLY": "1", "SKIP_ANALYSIS": "1", "DRY_RUN": "1"})
        _apply_settings(tmp)
        t0 = time.perf_counter()
        _ = process_devices_parallel(coll, sample_ids, start_utc, end_utc, analysis_dir)
        return (time.perf_counter() - t0) * 1000.0
    finally:
        os.environ.clear()
        os.environ.update(old_env)

def _auto_tune_if_needed(coll, start_utc: datetime, end_utc: datetime, analysis_dir: str, device_ids: List[str]) -> None:
    if os.environ.get("AUTO_TUNE", "0") != "1":
        return
    sample_n = int(os.environ.get("AUTO_TUNE_SAMPLE", "8"))
    sample_ids = device_ids[:sample_n]
    base = {
        "STREAMING_ENABLED": "0",
        "USE_INDEX_HINTS": "0",
        "THIN_PROJECTION": "1",
        "PREFERRED_INDEX": "ct_dev_asc",
        "MONGO_BATCH_SIZE": "500",
        "DECOMP_CHUNK_SIZE": "500",
        "DECOMP_EXECUTOR": "thread",
        "DECOMP_WORKERS": "1",
        "DEVICE_PARALLEL_WORKERS": "1",
        "PAGING_ENABLED": "0",
        "PAGE_SIZE": "500",
    }
    cand = {
        "STREAMING_ENABLED": "1",
        "USE_INDEX_HINTS": "1",
        "THIN_PROJECTION": "1",
        "PREFERRED_INDEX": "ct_dev_asc",
        "MONGO_BATCH_SIZE": "4000",
        "DECOMP_CHUNK_SIZE": "4000",
        "DECOMP_EXECUTOR": "thread",
        "DECOMP_WORKERS": str(max(1, (os.cpu_count() or 2))),
        "DEVICE_PARALLEL_WORKERS": str(max(1, min(12, (os.cpu_count() or 2))))
    }
    b_ms = _measure_read_time(coll, sample_ids, start_utc, end_utc, analysis_dir, base)
    c_ms = _measure_read_time(coll, sample_ids, start_utc, end_utc, analysis_dir, cand)
    improv = 1.0 if b_ms <= 0 else (b_ms - c_ms) / b_ms
    if improv < 0.30:
        fb = {
            "STREAMING_ENABLED": "1",
            "USE_INDEX_HINTS": "1",
            "THIN_PROJECTION": "1",
            "PREFERRED_INDEX": "ct_dev_asc",
            "MONGO_BATCH_SIZE": "2000",
            "DECOMP_CHUNK_SIZE": "2000",
            "DECOMP_EXECUTOR": "thread",
            "DECOMP_WORKERS": str(max(1, min(8, (os.cpu_count() or 2))))
        }
        _apply_settings(fb)
    else:
        _apply_settings(cand)
