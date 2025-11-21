import os
import re
import json
import csv
import argparse
from typing import Any, Set, Dict, Optional, List, Tuple
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import pymysql


def extract_ids_from_obj(obj: Any, ids: Set[str]):
    """递归提取对象中的 device_id/deviceId/deviceID 字段"""
    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                key_lower = k.lower()
                if key_lower in {"device_id", "deviceid"} and isinstance(v, str):
                    ids.add(v.strip())
                else:
                    extract_ids_from_obj(v, ids)
        elif isinstance(obj, list):
            for item in obj:
                extract_ids_from_obj(item, ids)
    except Exception:
        # 忽略递归过程中可能出现的类型或键错误
        pass


def parse_models_status(models_json_str: Optional[str]) -> Dict[str, Any]:
    result = {
        "heart_status": 0,
        "spine_status": 0,
        "lymph_status": 0,
        "pericardium_status": 0,
        "small_intestine_status": 0,
        "lung_status": 0,
        "stomach_status": 0,
    }
    if not models_json_str:
        return result
    def _norm(v: Any) -> int:
        try:
            return int(v)
        except Exception:
            return 0
    try:
        models_data = json.loads(models_json_str)

        synonyms = {
            "heart": {"心"},
            "spine": {"脊椎", "脊柱"},
            "lymph": {"淋巴"},
            "pericardium": {"心包"},
            "small_intestine": {"小肠"},
            "lung": {"肺"},
            "stomach": {"胃"},
        }

        organs_list = []
        if isinstance(models_data, list):
            organs_list = models_data
        elif isinstance(models_data, dict):
            if isinstance(models_data.get('organs'), list):
                organs_list = models_data['organs']
            else:
                organs_list = [models_data]

        for organ in organs_list:
            if not isinstance(organ, dict):
                continue
            name = organ.get("name")
            status = organ.get("status")
            if name in synonyms["heart"]:
                result["heart_status"] = _norm(status)
            elif name in synonyms["spine"]:
                result["spine_status"] = _norm(status)
            elif name in synonyms["lymph"]:
                result["lymph_status"] = _norm(status)
            elif name in synonyms["pericardium"]:
                result["pericardium_status"] = _norm(status)
            elif name in synonyms["small_intestine"]:
                result["small_intestine_status"] = _norm(status)
            elif name in synonyms["lung"]:
                result["lung_status"] = _norm(status)
            elif name in synonyms["stomach"]:
                result["stomach_status"] = _norm(status)

        if isinstance(models_data, dict):
            for k, v in models_data.items():
                kl = str(k).lower()
                if kl in {"heart", "spine", "lymph", "pericardium", "small_intestine", "lung", "stomach"}:
                    result[f"{kl}_status"] = _norm(v if not isinstance(v, dict) else v.get("status"))
    except (json.JSONDecodeError, TypeError):
        pass
    return result


def extract_first_value(obj: Any, key_variants: Set[str]) -> Optional[Any]:
    """递归提取对象中第一个匹配 key_variants 的字段值"""
    try:
        if isinstance(obj, dict):
            # 优先匹配顶层键
            for k, v in obj.items():
                if k.lower() in key_variants:
                    return v
            # 如果顶层没有，则递归搜索
            for k, v in obj.items():
                result = extract_first_value(v, key_variants)
                if result is not None:
                    return result
        elif isinstance(obj, list):
            for item in obj:
                result = extract_first_value(item, key_variants)
                if result is not None:
                    return result
    except Exception:
        pass
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取 device_id 并导出 wear/health 信息（加速版）")
    parser.add_argument(
        "--workers", type=int, default=max(4, (os.cpu_count() or 2) * 2),
        help="并发线程数，用于扫描与解析 JSON（默认 2xCPU）"
    )
    parser.add_argument(
        "--fast-text", action="store_true", default=True,
        help="启用快速文本扫描优先路径，尽量避免完整 JSON 解析"
    )
    parser.add_argument(
        "--skip-db", action="store_true", default=False,
        help="跳过 MySQL 查询，仅提取 device_id（最快）"
    )
    parser.add_argument(
        "--dirs", nargs="*", default=None,
        help="覆盖默认扫描目录列表，传入多个目录以扫描 JSON 文件"
    )
    parser.add_argument(
        "--csv-path", type=str, default=None,
        help="输出CSV文件路径，默认写入项目根目录"
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # 项目根目录为脚本上一级目录
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    target_dirs = args.dirs or [
        os.path.join(base_dir, 'output'),
        os.path.join(base_dir, 'risk_assessment_results'),
        os.path.join(base_dir, 'analysis_results'),
    ]

    device_ids: Set[str] = set()
    device_collect_times: Dict[str, str] = {}
    device_create_times: Dict[str, str] = {}

    def read_rr_summary_ids(summary_path: str) -> Tuple[Set[str], Dict[str, str]]:
        res: Set[str] = set()
        ctmap: Dict[str, str] = {}
        if not os.path.isfile(summary_path):
            return res, ctmap
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            items: List[Dict[str, Any]] = []
            if isinstance(data, dict):
                if isinstance(data.get('results'), list):
                    items = data.get('results')
                elif isinstance(data.get('success_files'), list):
                    items = data.get('success_files')
            for it in items:
                did = str((it or {}).get('device_id') or '').strip()
                ct = (it or {}).get('collect_time')
                if did:
                    res.add(did)
                    if isinstance(ct, str):
                        ctmap.setdefault(did, ct)
        except Exception:
            pass
        return res, ctmap

    rr_summary_path = os.path.join(base_dir, 'risk_results', 'advanced_7day_risk_batch_summary.json')
    ids_rr, ct_rr = read_rr_summary_ids(rr_summary_path)
    if ids_rr:
        for did in ids_rr:
            device_ids.add(did)
            if did in ct_rr and did not in device_collect_times:
                device_collect_times[did] = ct_rr[did]

    def build_rr_times(risk_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        cmap: Dict[str, str] = {}
        tmap: Dict[str, str] = {}
        if not os.path.isdir(risk_dir):
            return cmap, tmap
        try:
            for fn in os.listdir(risk_dir):
                if not fn.endswith('.json') or 'advanced_7day_risk_assessment' not in fn:
                    continue
                fp = os.path.join(risk_dir, fn)
                try:
                    with open(fp, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    did = str((data or {}).get('device_id') or '').strip()
                    ct = (data or {}).get('collect_time')
                    at = (data or {}).get('analysis_timestamp')
                    if did:
                        if isinstance(ct, str):
                            cmap.setdefault(did, ct)
                        if isinstance(at, str):
                            tmap.setdefault(did, at)
                except Exception:
                    continue
        except Exception:
            pass
        return cmap, tmap

    risk_dir = os.path.join(base_dir, 'risk_results')
    collect_map2, create_map2 = build_rr_times(risk_dir)
    for k, v in collect_map2.items():
        device_collect_times.setdefault(k, v)
    for k, v in create_map2.items():
        device_create_times.setdefault(k, v)

    # 1) 从 output 文件名提取（形如 <device_id>_YYYYMMDD_HHMMSS.json）
    out_dir = os.path.join(base_dir, 'output')
    if os.path.isdir(out_dir):
        # 使用 scandir 更快地遍历目录项
        with os.scandir(out_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                fn = entry.name
                if fn.endswith('.json'):
                    m = re.match(r'^([0-9]+)_', fn)
                    if m:
                        device_ids.add(m.group(1))

    # 2) 从 JSON 内容提取 device_id 和 collect_time
    # 预编译正则（用于快速文本扫描）
    id_pat = re.compile(r'"(device_id|deviceId|deviceID)"\s*:\s*("?)([0-9]+)\2', re.IGNORECASE)
    ct_pat = re.compile(r'"collect_time"\s*:\s*"([^"]+)"')

    def process_file(fp: str) -> Tuple[Set[str], Optional[str]]:
        ids_found: Set[str] = set()
        ct_val: Optional[str] = None
        try:
            if args.fast_text:
                # 快速文本扫描优先路径：避免完整 JSON 解析
                with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read()
                for m in id_pat.finditer(txt):
                    ids_found.add(m.group(3))
                mct = ct_pat.search(txt)
                if mct:
                    ct_val = mct.group(1)
                # 如果已找到 device_id，则不再解析 JSON
                if ids_found:
                    return ids_found, ct_val
            # 回退到完整 JSON 解析（确保稳健性）
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            extract_ids_from_obj(data, ids_found)
            if ids_found:
                ct = extract_first_value(data, {"collect_time"})
                if isinstance(ct, str):
                    ct_val = ct
        except Exception:
            pass
        return ids_found, ct_val

    # 并发扫描所有 JSON 文件
    json_files: List[str] = []
    for d in target_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fn in files:
                if fn.endswith('.json'):
                    json_files.append(os.path.join(root, fn))

    if json_files:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_file, fp): fp for fp in json_files}
            for fut in as_completed(futures):
                ids_found, ct_val = fut.result()
                if not ids_found:
                    continue
                for did in ids_found:
                    device_ids.add(did)
                    if ct_val and did not in device_collect_times:
                        device_collect_times[did] = ct_val

    # 排序并输出
    sorted_ids = sorted(device_ids)

    # 写入到根目录文件，便于后续使用
    txt_path = os.path.join(base_dir, 'device_ids.txt')
    json_path = os.path.join(base_dir, 'device_ids.json')
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            for did in sorted_ids:
                f.write(did + '\n')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_ids, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print(f"共提取到 {len(sorted_ids)} 个 device_id。")

    # ===== MySQL 查询部分 =====
    # 配置（按用户提供）
    MYSQL_HOST = "rr-2ze57t6e5586l181hyo.mysql.rds.aliyuncs.com"
    MYSQL_PORT = 3306
    MYSQL_USER = "ppg_reader"
    MYSQL_PASSWORD = "Bm*PiyeQjD6cGii"

    # 北京时区，计算昨天日期（用于 andun_health.h_health_analysis 的 source_date 过滤）
    beijing_tz = timezone(timedelta(hours=8))
    yesterday_str = (datetime.now(beijing_tz) - timedelta(days=1)).strftime("%Y-%m-%d")

    def get_conn(db_name: str) -> Optional[pymysql.connections.Connection]:
        try:
            return pymysql.connect(
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=db_name,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=8,
                read_timeout=15,
                write_timeout=15,
            )
        except Exception as e:
            print(f"连接数据库 {db_name} 失败: {e}")
            return None

    def resolve_order_column(conn: pymysql.connections.Connection, db_name: str, table_name: str,
                              candidates: List[str]) -> Tuple[Optional[str], List[str]]:
        """检测候选排序列在表中是否存在，返回第一个可用列及存在的列列表"""
        exists: List[str] = []
        if conn is None:
            return None, exists
        try:
            placeholders = ",".join(["%s"] * len(candidates))
            sql = (
                "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s AND COLUMN_NAME IN (" + placeholders + ")"
            )
            with conn.cursor() as cur:
                cur.execute(sql, [db_name, table_name] + candidates)
                rows = cur.fetchall() or []
                exists = [r["COLUMN_NAME"] for r in rows]
        except Exception:
            # 如果无法查询信息架构，返回空列表
            exists = []
        for c in candidates:
            if c in exists:
                return c, exists
        return None, exists

    # 可选：建立数据库连接
    conn_watch = None
    conn_health = None
    if not args.skip_db:
        conn_watch = get_conn("andun_watch")
        conn_health = get_conn("andun_health")

    # 预检测 andun_watch.user_wear_active_data 的可用排序列
    order_candidates = ["update_time", "create_time", "modify_time", "id"]
    order_col, available_cols = resolve_order_column(conn_watch, "andun_watch", "user_wear_active_data", order_candidates) if conn_watch else (None, [])
    if conn_watch and not order_col:
        # 回退策略：优先 create_time，其次 id；若都不可用，则不排序仅取一条
        if "create_time" in available_cols:
            order_col = "create_time"
        elif "id" in available_cols:
            order_col = "id"

    def fetch_latest_wear_user_id(conn: pymysql.connections.Connection, device_id: str) -> Optional[str]:
        """在 andun_watch.user_wear_active_data 中按检测到的列倒序取最近 wear_user_id"""
        if conn is None:
            return None
        try:
            with conn.cursor() as cur:
                if order_col:
                    sql = f"SELECT wear_user_id FROM user_wear_active_data WHERE device_id=%s ORDER BY `{order_col}` DESC LIMIT 1"
                    cur.execute(sql, (device_id,))
                else:
                    sql = "SELECT wear_user_id FROM user_wear_active_data WHERE device_id=%s LIMIT 1"
                    cur.execute(sql, (device_id,))
                row = cur.fetchone()
                if row and row.get("wear_user_id") is not None:
                    return str(row["wear_user_id"])  # 统一为字符串
        except Exception as e:
            # 简化输出，避免无用的列错误噪声
            print(f"查询 wear_user_id 失败(device_id={device_id}): {e}")
        return None

    def bulk_fetch_latest_wear_user_ids(conn: pymysql.connections.Connection, device_list: List[str]) -> Dict[str, str]:
        """批量获取最近 wear_user_id，减少 N 次往返查询"""
        result: Dict[str, str] = {}
        if conn is None or not device_list:
            return result
        try:
            with conn.cursor() as cur:
                # 分块执行，避免 SQL 占位符过多
                CHUNK = 500
                for i in range(0, len(device_list), CHUNK):
                    chunk = device_list[i:i+CHUNK]
                    placeholders = ",".join(["%s"] * len(chunk))
                    if order_col:
                        sql = (
                            f"SELECT t.device_id, t.wear_user_id FROM user_wear_active_data t "
                            f"JOIN (SELECT device_id, MAX(`{order_col}`) AS mx FROM user_wear_active_data "
                            f"      WHERE device_id IN ({placeholders}) GROUP BY device_id) m "
                            f"ON t.device_id=m.device_id AND t.`{order_col}`=m.mx"
                        )
                    else:
                        # 无可用排序列时，退化为任意一条（可能不是最新）
                        sql = (
                            f"SELECT device_id, wear_user_id FROM user_wear_active_data "
                            f"WHERE device_id IN ({placeholders}) GROUP BY device_id"
                        )
                    cur.execute(sql, chunk)
                    rows = cur.fetchall() or []
                    for r in rows:
                        did = str(r.get("device_id"))
                        wid = r.get("wear_user_id")
                        if did and wid is not None:
                            result[did] = str(wid)
        except Exception as e:
            print(f"批量查询 wear_user_id 失败: {e}")
        return result

    def fetch_health_analysis_for_day(conn: pymysql.connections.Connection, wear_user_id: str, day_str: str) -> Optional[Dict[str, Any]]:
        """在 andun_health.h_health_analysis 中按 create_time 取当天最新记录"""
        if conn is None:
            return None
        sql = (
            "SELECT T_WEAR_USER_ID, source_date, create_time, icon , models "
            "FROM h_health_analysis "
            "WHERE T_WEAR_USER_ID=%s AND DATE(source_date)=%s "
            "ORDER BY source_date DESC LIMIT 1"
        )
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (wear_user_id, day_str))
                row = cur.fetchone()
                return row if row else None
        except Exception as e:
            print(f"查询 h_health_analysis 失败(wear_user_id={wear_user_id}, day={day_str}): {e}")
            return None

    def fetch_latest_health_analysis(conn: pymysql.connections.Connection, wear_user_id: str) -> Optional[Dict[str, Any]]:
        """当按天无记录时，回退为获取该用户最新一条健康分析记录"""
        if conn is None:
            return None
        sql = (
            "SELECT T_WEAR_USER_ID, source_date, create_time, icon , models "
            "FROM h_health_analysis "
            "WHERE T_WEAR_USER_ID=%s "
            "ORDER BY source_date DESC, create_time DESC LIMIT 1"
        )
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (wear_user_id,))
                row = cur.fetchone()
                return row if row else None
        except Exception as e:
            print(f"查询 h_health_analysis 最新记录失败(wear_user_id={wear_user_id}): {e}")
            return None


    # 解析风险等级映射函数与文件
    def parse_risk_levels(summary_path: str) -> Dict[str, Tuple[str, str]]:
        """读取批量汇总文件，解析 device_id 对应的 mi/stroke 风险等级。
        支持两种结构：{"results": [...]} 或 {"success_files": [...]}。
        """
        result: Dict[str, Tuple[str, str]] = {}
        if not os.path.isfile(summary_path):
            return result
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            items: List[Dict[str, Any]] = []
            if isinstance(data, dict):
                if isinstance(data.get("results"), list):
                    items = data.get("results")
                elif isinstance(data.get("success_files"), list):
                    items = data.get("success_files")
            for it in items:
                did = str((it or {}).get("device_id") or "").strip()
                mi = (it or {}).get("mi_risk_level")
                stroke = (it or {}).get("stroke_risk_level")
                if did:
                    result[did] = (mi or "", stroke or "")
        except Exception as e:
            print(f"读取风险汇总失败({summary_path}): {e}")
        return result

    # 风险等级1来源：risk_assessment_results + 漏报批量汇总
    ra_summary_path = os.path.join(base_dir, 'risk_assessment_results', 'batch_processing_summary.json')
    leak_summary_path = os.path.join(base_dir, '漏报数据', '漏报风险数据', '601012202069926', '心脑1', 'batch_processing_summary.json')
    risk_level1_map = parse_risk_levels(ra_summary_path)
    leak_map = parse_risk_levels(leak_summary_path)
    for k, v in leak_map.items():
        risk_level1_map.setdefault(k, v)

    # 风险等级2来源：risk_results 高级7天批量汇总 + 单设备文件回退
    rr_summary_path = os.path.join(base_dir, 'risk_results', 'advanced_7day_risk_batch_summary.json')
    risk_level2_map = parse_risk_levels(rr_summary_path)

    def build_risk_level2_fallback(risk_dir: str) -> Dict[str, Tuple[str, str]]:
        fallback: Dict[str, Tuple[str, str]] = {}
        if not os.path.isdir(risk_dir):
            return fallback
        try:
            for fn in os.listdir(risk_dir):
                if not fn.endswith('.json') or 'advanced_7day_risk_assessment' not in fn:
                    continue
                fp = os.path.join(risk_dir, fn)
                try:
                    with open(fp, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    did = str((data or {}).get('device_id') or '').strip()
                    rp = (data or {}).get('risk_prediction') or {}
                    mi = (rp.get('myocardial_infarction') or {}).get('7_day_risk_level') or (rp.get('myocardial_infarction') or {}).get('risk_level')
                    stroke = (rp.get('stroke') or {}).get('7_day_risk_level') or (rp.get('stroke') or {}).get('risk_level')
                    if did:
                        fallback.setdefault(did, (mi or '', stroke or ''))
                except Exception:
                    continue
        except Exception:
            pass
        return fallback

    risk_dir = os.path.join(base_dir, 'risk_results')
    fallback_map = build_risk_level2_fallback(risk_dir)
    for k, v in fallback_map.items():
        risk_level2_map.setdefault(k, v)

    export_rows: List[Dict[str, Any]] = []

    # 如果跳过数据库，直接生成空 wear_user_id 与 health 字段
    wear_map: Dict[str, str] = {}
    if not args.skip_db:
        wear_map = bulk_fetch_latest_wear_user_ids(conn_watch, sorted_ids)

    for did in sorted_ids:
        wear_user_id = None if args.skip_db else wear_map.get(did) or fetch_latest_wear_user_id(conn_watch, did)
        
        query_date_str = yesterday_str
        collect_time = device_collect_times.get(did)
        if collect_time:
            try:
                # 规范化为 YYYY-MM-DD
                # 支持 ISO 格式时间戳 (e.g., "2023-10-27T10:00:00Z") 或 日期字符串 ("2023-10-27")
                if 'T' in collect_time:
                    # 假设是带 'T' 的 ISO 格式
                    query_date_str = datetime.fromisoformat(collect_time.replace('Z', '+00:00')).strftime('%Y-%m-%d')
                elif ' ' in collect_time:
                    # 假设是带空格的 'YYYY-MM-DD HH:MM:SS' 格式
                    query_date_str = datetime.strptime(collect_time, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
                else:
                    # 否则，尝试解析为 'YYYY-MM-DD'
                    query_date_str = datetime.strptime(collect_time, '%Y-%m-%d').strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                print(f"警告: device_id={did} 的 collect_time '{collect_time}' 格式无法解析，将使用昨天日期。")

        ha = None if args.skip_db else (fetch_health_analysis_for_day(conn_health, wear_user_id, query_date_str) if wear_user_id else None)
        # 如果当天没有记录，则回退到该用户的最新健康分析记录
        if not args.skip_db and wear_user_id and not ha:
            ha = fetch_latest_health_analysis(conn_health, wear_user_id)

        # 优先使用健康表中的 T_WEAR_USER_ID 写入 CSV（更权威）
        wear_user_id_out = str(((ha or {}).get("T_WEAR_USER_ID")) or (wear_user_id or ""))

        mi1, stroke1 = risk_level1_map.get(did, ("", ""))
        mi2, stroke2 = risk_level2_map.get(did, ("", ""))
        status_map = parse_models_status((ha or {}).get("models"))
        hs = status_map.get("heart_status", 0)
        ls = status_map.get("lymph_status", 0)
        ps = status_map.get("pericardium_status", 0)
        si = status_map.get("small_intestine_status", 0)
        ss = status_map.get("spine_status", 0)
        lg = status_map.get("lung_status", 0)
        st = status_map.get("stomach_status", 0)

        def _lvl_to_status(s: Optional[str]) -> int:
            if not isinstance(s, str):
                return 0
            if '高' in s:
                return 3
            if '中' in s:
                return 2
            if '低' in s:
                return 1
            return 0

        if hs == 0 and ls == 0 and ps == 0 and si == 0 and ss == 0 and lg == 0 and st == 0:
            mi_status = _lvl_to_status(mi2 or mi1)
            stroke_status = _lvl_to_status(stroke2 or stroke1)
            hs = mi_status
            ls = mi_status
            ps = mi_status
            si = mi_status
            ss = stroke_status
            lg = stroke_status
            st = stroke_status

        mi_val = {0: 0.0, 1: 10.0, 2: 20.0, 3: 40.0}.get(hs, 0.0)
        lymph_val = {0: 0.0, 1: 10.0, 2: 20.0, 3: 40.0}.get(ls, 0.0)
        pericardium_mi_val = {0: 0.0, 1: 2.5, 2: 5.0, 3: 10.0}.get(ps, 0.0)
        intestine_val = {0: 0.0, 1: 2.5, 2: 5.0, 3: 10.0}.get(si, 0.0)
        sys_mi_risk_score = mi_val + lymph_val + pericardium_mi_val + intestine_val

        spine_val = {0: 0.0, 1: 12.5, 2: 15.0, 3: 30.0}.get(ss, 0.0)
        pericardium_stroke_val = {0: 0.0, 1: 7.5, 2: 15.0, 3: 30.0}.get(ps, 0.0)
        lung_val = {0: 0.0, 1: 6.25, 2: 12.5, 3: 25.0}.get(lg, 0.0)
        stomach_val = {0: 0.0, 1: 3.25, 2: 7.5, 3: 15.0}.get(st, 0.0)
        sys_stroke_risk_score = spine_val + pericardium_stroke_val + lung_val + stomach_val

       # sys_score = sys_mi_risk_score + sys_stroke_risk_score

        src_date_out = (ha or {}).get("source_date") or device_collect_times.get(did) or ""
        try:
            day = _norm_day_str(src_date_out)
            src_date_out = day or query_date_str
            if src_date_out != query_date_str:
                src_date_out = query_date_str
        except Exception:
            src_date_out = query_date_str

        create_time_out = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        export_rows.append({
            "device_id": did,
            "wear_user_id": wear_user_id_out,
            "source_date": src_date_out,
            "create_time": create_time_out,
            "heart_status": hs,
            "spine_status": ss,
            "lymph_status": ls,
            "pericardium_status": ps,
            "small_intestine_status": si,
            "lung_status": lg,
            "stomach_status": st,
            "sys_mi_risk_score": sys_mi_risk_score,
            "sys_stroker_risk_score": sys_stroke_risk_score,
            # "sys_score": sys_score,
            "mi_risk_level2": mi2,
            "stroke_risk_level2": stroke2,
        })

    # 导出到当前文件夹（项目根目录）
    csv_path = args.csv_path or os.path.join(base_dir, "device_wear_health_export.csv")
    try:
        fieldnames = [
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
            # "sys_score",
            "mi_risk_level2",
            "stroke_risk_level2",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(export_rows)
        print(f"CSV 已生成: {csv_path} (共 {len(export_rows)} 行，过滤日期: {yesterday_str})")
    except PermissionError:
        alt_dir = os.path.dirname(csv_path) or base_dir
        alt_path = os.path.join(alt_dir, f"device_wear_health_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        try:
            fieldnames = [
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
               # "sys_score",
                "mi_risk_level2",
                "stroke_risk_level2",
            ]
            with open(alt_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(export_rows)
            print(f"CSV 因权限占用，已写入: {alt_path} (共 {len(export_rows)} 行，过滤日期: {yesterday_str})")
        except Exception as e2:
            print(f"写入 CSV 失败(备选路径): {e2}")
    except Exception as e:
        print(f"写入 CSV 失败: {e}")

    # 关闭连接
    try:
        if conn_watch:
            conn_watch.close()
        if conn_health:
            conn_health.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
def _norm_day_str(val: Any) -> Optional[str]:
    try:
        if isinstance(val, datetime):
            return val.strftime('%Y-%m-%d')
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return None
            if 'T' in s:
                return datetime.fromisoformat(s.replace('Z', '+00:00')).strftime('%Y-%m-%d')
            if ' ' in s:
                return datetime.strptime(s, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
            try:
                return datetime.strptime(s, '%Y-%m-%d').strftime('%Y-%m-%d')
            except Exception:
                return None
    except Exception:
        return None
    return None