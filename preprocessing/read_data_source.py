from datetime import datetime, timedelta
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple

from tingis.global_vars import TING_DATABASE_CONFIG, hotline_online_datasource_table_name
from tingis.knowledge_base.ob_db_manager.ob_db_operations import OBDatabaseManager
from tingis.utils.common_utils import BEIJING_TZ, str_to_timestamp


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s: %(lineno)d - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_ting_tiyan(db_manager, start_datetime, end_datetime, table=None):
    if table is None:
        table = hotline_online_datasource_table_name

    # 读取用户原声摘要（客权体验）

    column = 'gmt_create'
    sql = f"SELECT * FROM {table} WHERE {column} >= %s AND {column} < %s"
    query_results = db_manager.execute_query(sql, (start_datetime, end_datetime))

    if query_results is None:
        logger.warning("No query results returned.")
        return []

    processed_results = []
    for i in query_results:
        try:
            add_date_str = i['add_date'].strftime("%Y-%m-%d %H:%M:%S")
            gmt_create_str = i['gmt_create'].strftime("%Y-%m-%d %H:%M:%S")
            timestamp = str_to_timestamp(add_date_str)
            processed_results.append(
                {
                    'voice_id': i['voice_id'],
                    'voice_title': i['title'],
                    'metadata': {
                        'gmt_create': gmt_create_str,
                        'add_date': add_date_str,
                        'voice_channel': i['voice_channel'],
                        'uid': i['uid'],
                        'voice_detail_link': i['voice_detail_link'],
                    },
                    '_timestamp': timestamp
                }
            )
        except Exception as e:
            logger.warning(f"Error processing voice item: {i.get('voice_id', 'unknown')}, error: {e}")
            continue

    # 排序逻辑：按 add_date 升序
    processed_results.sort(key=lambda x: x['_timestamp'])

    # 清理临时字段
    for item in processed_results:
        del item['_timestamp']

    logger.info(f'Number of query results: {len(processed_results)}')
    return processed_results

if __name__ == '__main__':

    # 初始化资源
    ob_db_manager = OBDatabaseManager(TING_DATABASE_CONFIG)
    query_time_window = 60

    start_time_str = "2025-11-06 11:00:00"
    start_time_obj = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    end_time_obj = start_time_obj + timedelta(seconds=query_time_window)
    start_time_str = start_time_obj.strftime("%Y-%m-%d %H:%M:%S")
    end_time_str = end_time_obj.strftime("%Y-%m-%d %H:%M:%S")

    # 1. 查询
    query_start = datetime.now(BEIJING_TZ)
    voices_info_raw = fetch_ting_tiyan(ob_db_manager, start_time_str, end_time_str, hotline_online_datasource_table_name)
    # voices_info_raw, new_max_id = fetch_ting_tiyan_v3(ob_db_manager, start_time = start_time_str, is_initial = True)
    query_latency = (datetime.now(BEIJING_TZ) - query_start).total_seconds()
    logger.info(f"[Query] 时间范围: {start_time_str} ~ {end_time_str} | 总数: {len(voices_info_raw) if voices_info_raw else 0} | 耗时: {query_latency:.2f}s")
    # print(voices_info_raw[0:10])