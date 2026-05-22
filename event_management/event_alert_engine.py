# tingis/event_management/event_alert_engine.py

import time
import logging
from typing import List, Tuple, Optional
import numpy as np
from tingis.utils.common_utils import timestamp_to_str
from tingis.knowledge_base.ob_db_manager.ob_db_operations import OBDatabaseManager
from tingis.global_vars import (
    risk_event_volume_timeline_table_name,
    ALERTING_MIN_INCREMENTAL_VOLUME,
    SPECIAL_ALERTING_THRESHOLDS,
    ALERTING_WINDOW_HOURS,
    DYNAMIC_BASELINE_HISTORICAL_DAYS,
    DYNAMIC_BASELINE_MIN_HISTORICAL_SAMPLES,
    DYNAMIC_BASELINE_MINIMUM_STD_DEV,
)

logger = logging.getLogger(__name__)

class EventAlertEngine:
    def __init__(self, ob_db_manager: OBDatabaseManager):
        self.db = ob_db_manager
        # 全局阈值
        self.default_min_incremental = ALERTING_MIN_INCREMENTAL_VOLUME
        # 特殊阈值
        self.special_thresholds = SPECIAL_ALERTING_THRESHOLDS

        self.historical_days = DYNAMIC_BASELINE_HISTORICAL_DAYS
        self.min_historical_samples = DYNAMIC_BASELINE_MIN_HISTORICAL_SAMPLES
        self.alert_window_seconds = ALERTING_WINDOW_HOURS * 3600

    def set_current_time(self, ts: int):
        self.current_ts = ts

    def _get_threshold_for_event(self, gm_code: Optional[str]) -> int:
        """
        根据 gm_code 获取增量声量阈值。
        注：移除了总声量阈值，因为 total_volume >= incremental_volume 恒成立，
        当阈值相等时总声量检查无实际作用。
        """
        incremental_threshold = self.default_min_incremental

        if gm_code and gm_code in self.special_thresholds:
            incremental_threshold = self.special_thresholds[gm_code]

        return incremental_threshold

    def should_alert_with_dynamic_baseline(
        self,
        event_id: str,
        gm_code: Optional[str],
        current_total_volume: int,
        current_incremental_volume: int
    ) -> Tuple[bool, dict]:
        """
        判断是否应告警，基于增量声量的动态基线，并返回决策上下文。

        1. 使用 mean + 2*std
        2. 历史数据不足时，仅依赖静态阈值判断是否告警
        3. 引入最小标准差（基础噪声）来处理历史数据为常量（如全0）的情况。
        4. 保证最终的动态阈值不低于静态阈值。

        返回: (是否告警, 决策上下文词典)
        """
        # Step 1: 获取静态阈值
        min_incremental_threshold = self._get_threshold_for_event(gm_code)

        context = {
            'check_timestamp': self.current_ts,
            'event_id': event_id,
            'gm_code': gm_code,
            'current_total_volume': current_total_volume,
            'current_incremental_volume': current_incremental_volume,
            'min_incremental_volume_threshold': min_incremental_threshold,
            'alert_window_hours': ALERTING_WINDOW_HOURS
        }

        # Step 2: 基础静态阈值检查(活跃度门槛)
        # 近期增量声量：近期有明显的活跃度（eg. 过去1小时新增原声>=3）
        if current_incremental_volume < min_incremental_threshold:
            context['decision_reason'] = f"Failed basic incremental volume threshold. (inc_vol: {current_incremental_volume} < {min_incremental_threshold})."
            return False, context

        # Step 3: 获取历史数据
        historical_incremental_volumes = self._get_historical_incremental_volumes(event_id)
        context['historical_samples_count'] = len(historical_incremental_volumes)
        context['historical_incremental_volumes'] = historical_incremental_volumes

        # Step 4: 决策逻辑
        # 情况A: 历史样本不足，无法计算动态基线
        if len(historical_incremental_volumes) < self.min_historical_samples:
            context['decision_reason'] = (f"Insufficient historical data (found {len(historical_incremental_volumes)}, need {self.min_historical_samples}). "
                                        f"Dynamic baseline skipped. Alert is triggered because static threshold was met.")
            context['final_decision_threshold'] = min_incremental_threshold
            return True, context

        # 情况B: 历史样本充足，计算动态基线
        mean_vol = sum(historical_incremental_volumes) / len(historical_incremental_volumes)

        # 计算原始标准差
        if len(historical_incremental_volumes) > 1:
            std_vol_raw = (sum((x - mean_vol) ** 2 for x in historical_incremental_volumes) / (len(historical_incremental_volumes) - 1)) ** 0.5
        else:
            std_vol_raw = 0

        # 引入基础噪声，确保标准差不为0或过小
        effective_std_vol = max(std_vol_raw, DYNAMIC_BASELINE_MINIMUM_STD_DEV)

        # 计算原始动态阈值
        dynamic_threshold_raw = mean_vol + 2 * effective_std_vol

        # 确保最终阈值不低于静态阈值
        final_decision_threshold = max(dynamic_threshold_raw, min_incremental_threshold)

        # 填充决策上下文，便于调试
        context['historical_mean'] = round(mean_vol, 2)
        context['historical_std_dev'] = round(std_vol_raw, 2)
        context['effective_std_dev'] = round(effective_std_vol, 2)
        context['dynamic_threshold'] = round(dynamic_threshold_raw, 2)
        context['final_decision_threshold'] = round(final_decision_threshold, 2)

        is_abnormal = current_incremental_volume > final_decision_threshold

        if is_abnormal:
            context['decision_reason'] = f"Current incremental volume ({current_incremental_volume}) exceeded final decision threshold ({context['final_decision_threshold']:.2f})."
        else:
            context['decision_reason'] = f"Current incremental volume ({current_incremental_volume}) did not exceed final decision threshold ({context['final_decision_threshold']:.2f})."

        return is_abnormal, context

    def _get_historical_incremental_volumes(self, event_id: str) -> List[int]:
        """
        获取历史同期【增量】声量。
        对于过去N天的每一天，汇总其在同一时间窗口内的 incremental_volume_at_snapshot
        """
        historical_volumes = []

        for days_ago in range(1, self.historical_days + 1):
            window_end_ts = self.current_ts - days_ago * 86400
            window_start_ts = window_end_ts - self.alert_window_seconds

            start_str = timestamp_to_str(window_start_ts)
            end_str = timestamp_to_str(window_end_ts)

            sql = f"""SELECT SUM(incremental_volume_at_snapshot) as total_increment
                      FROM {risk_event_volume_timeline_table_name}
                      WHERE risk_event_id = %s
                      AND snapshot_time > %s
                      AND snapshot_time <= %s
                   """

            result = self.db.execute_query(sql, (event_id, start_str, end_str))

            if result and result[0]['total_increment'] is not None:
                historical_volumes.append(int(result[0]['total_increment']))
            else:
                historical_volumes.append(0)

        return historical_volumes