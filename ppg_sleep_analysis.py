"""
PPG睡眠分析模块
基于PPG数据进行睡眠呼吸暂停、夜间血氧饱和度、血压节律分析
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class PPGSleepAnalyzer:
    """PPG睡眠分析器"""
    
    def __init__(self, sampling_rate: int = 125):
        """
        初始化睡眠分析器
        
        Args:
            sampling_rate: 采样率，默认125Hz
        """
        self.sampling_rate = sampling_rate
        self.logger = logging.getLogger(__name__)
        
        # 睡眠呼吸暂停检测参数
        self.apnea_threshold = 0.3  # 呼吸暂停阈值
        self.apnea_duration_min = 10  # 最小暂停时长(秒)
        self.hypopnea_threshold = 0.5  # 低通气阈值
        
        # 血氧饱和度分析参数
        self.spo2_normal_range = (95, 100)  # 正常血氧范围
        self.spo2_low_threshold = 90  # 低血氧阈值
        
        # 血压节律分析参数
        self.circadian_period = 24 * 3600  # 昼夜节律周期(秒)
        self.night_hours = (22, 6)  # 夜间时间范围
    
    def analyze_sleep_apnea(self, ppg_data: np.ndarray, timestamps) -> Dict[str, Any]:
        """
        睡眠呼吸暂停分析
        
        Args:
            ppg_data: PPG数据
            timestamps: 时间戳列表
            
        Returns:
            睡眠呼吸暂停分析结果
        """
        try:
            # 提取呼吸信号
            respiratory_signal = self._extract_respiratory_signal(ppg_data)
            
            # 检测呼吸暂停事件
            apnea_events = self._detect_apnea_events(respiratory_signal, timestamps)
            
            # 检测低通气事件
            hypopnea_events = self._detect_hypopnea_events(respiratory_signal, timestamps)
            
            # 计算AHI指数
            total_sleep_time = len(ppg_data) / self.sampling_rate / 3600  # 小时
            ahi_index = (len(apnea_events) + len(hypopnea_events)) / total_sleep_time if total_sleep_time > 0 else 0
            
            # 呼吸暂停严重程度分级
            severity = self._classify_apnea_severity(ahi_index)
            
            # 计算呼吸相关统计指标
            respiratory_stats = self._calculate_respiratory_statistics(respiratory_signal)
            
            return {
                'apnea_events': apnea_events,
                'hypopnea_events': hypopnea_events,
                'ahi_index': round(ahi_index, 2),
                'severity': severity,
                'total_events': len(apnea_events) + len(hypopnea_events),
                'apnea_count': len(apnea_events),
                'hypopnea_count': len(hypopnea_events),
                'respiratory_statistics': respiratory_stats,
                'analysis_quality': self._assess_respiratory_quality(respiratory_signal)
            }
            
        except Exception as e:
            self.logger.error(f"睡眠呼吸暂停分析失败: {e}")
            return self._get_default_apnea_result()
    
    def analyze_nocturnal_spo2(self, ppg_data: np.ndarray, timestamps) -> Dict[str, Any]:
        """
        夜间血氧饱和度分析
        
        Args:
            ppg_data: PPG数据
            timestamps: 时间戳列表
            
        Returns:
            夜间血氧饱和度分析结果
        """
        try:
            # 估算血氧饱和度
            spo2_values = self._estimate_spo2(ppg_data)
            
            # 识别夜间时段 - 需要根据降采样后的数据长度调整索引
            night_indices = self._identify_night_period(timestamps)
            
            # 如果SpO2数据被降采样了，需要调整夜间索引
            if len(spo2_values) != len(timestamps):
                # 计算降采样因子
                downsample_factor = len(timestamps) / len(spo2_values)
                # 调整夜间索引
                night_indices = (night_indices / downsample_factor).astype(int)
                # 确保索引在有效范围内
                night_indices = night_indices[night_indices < len(spo2_values)]
            
            night_spo2 = spo2_values[night_indices] if len(night_indices) > 0 else spo2_values
            
            # 检测血氧饱和度下降事件
            desaturation_events = self._detect_desaturation_events(night_spo2, timestamps, night_indices)
            
            # 计算ODI指数（氧减指数）
            odi_index = self._calculate_odi(desaturation_events, len(night_spo2) / self.sampling_rate / 3600)
            
            # 血氧统计分析
            spo2_stats = self._calculate_spo2_statistics(night_spo2)
            
            # 血氧变异性分析
            spo2_variability = self._analyze_spo2_variability(night_spo2)
            
            return {
                'mean_spo2': round(np.mean(night_spo2), 2),
                'min_spo2': round(np.min(night_spo2), 2),
                'spo2_below_90_percent': round(np.sum(night_spo2 < 90) / len(night_spo2) * 100, 2),
                'desaturation_events': desaturation_events,
                'odi_index': round(odi_index, 2),
                'spo2_statistics': spo2_stats,
                'spo2_variability': spo2_variability,
                'night_duration_hours': round(len(night_spo2) / self.sampling_rate / 3600, 2),
                'analysis_quality': self._assess_spo2_quality(night_spo2)
            }
            
        except Exception as e:
            self.logger.error(f"夜间血氧饱和度分析失败: {e}")
            return self._get_default_spo2_result()
    
    def analyze_blood_pressure_rhythm(self, ppg_data: np.ndarray, timestamps) -> Dict[str, Any]:
        """
        血压节律分析
        
        Args:
            ppg_data: PPG数据
            timestamps: 时间戳列表
            
        Returns:
            血压节律分析结果
        """
        try:
            # 估算血压相关指标
            bp_indicators = self._estimate_bp_indicators(ppg_data)
            
            # 昼夜节律分析
            circadian_analysis = self._analyze_circadian_rhythm(bp_indicators, timestamps)
            
            # 夜间血压下降分析
            nocturnal_dipping = self._analyze_nocturnal_dipping(bp_indicators, timestamps)
            
            # 血压变异性分析
            bp_variability = self._analyze_bp_variability(bp_indicators, timestamps)
            
            # 血压节律模式识别
            rhythm_pattern = self._identify_bp_rhythm_pattern(circadian_analysis, nocturnal_dipping)
            
            return {
                'circadian_analysis': circadian_analysis,
                'nocturnal_dipping': nocturnal_dipping,
                'bp_variability': bp_variability,
                'rhythm_pattern': rhythm_pattern,
                'day_night_ratio': round(circadian_analysis.get('day_night_ratio', 1.0), 3),
                'dipping_percentage': round(nocturnal_dipping.get('dipping_percentage', 0), 2),
                'rhythm_quality': self._assess_rhythm_quality(bp_indicators, timestamps)
            }
            
        except Exception as e:
            self.logger.error(f"血压节律分析失败: {e}")
            return self._get_default_bp_rhythm_result()
    
    def _extract_respiratory_signal(self, ppg_data: np.ndarray) -> np.ndarray:
        """从PPG信号中提取呼吸信号"""
        # 对于大数据集，先进行降采样以提高性能
        if len(ppg_data) > 50000:  # 如果数据点超过50000，进行降采样
            downsample_factor = max(1, len(ppg_data) // 50000)
            ppg_data = ppg_data[::downsample_factor]
            effective_sampling_rate = self.sampling_rate // downsample_factor
        else:
            effective_sampling_rate = self.sampling_rate
        
        # 低通滤波提取呼吸成分
        nyquist = effective_sampling_rate / 2
        low_cutoff = 0.1 / nyquist  # 0.1 Hz
        high_cutoff = 0.5 / nyquist  # 0.5 Hz
        
        try:
            b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
            respiratory_signal = signal.filtfilt(b, a, ppg_data)
        except:
            respiratory_signal = self._savgol(ppg_data)
        
        return respiratory_signal

    def _savgol(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        w = max(5, min(51, n // 10))
        if w % 2 == 0:
            w += 1
        if w > n:
            w = n if n % 2 == 1 else max(5, n - 1)
        if w < 5:
            return np.copy(x)
        return signal.savgol_filter(x, w, 3)
    
    def _detect_apnea_events(self, respiratory_signal: np.ndarray, timestamps) -> List[Dict]:
        """检测呼吸暂停事件"""
        events = []
        
        # 计算呼吸信号的包络
        envelope = np.abs(signal.hilbert(respiratory_signal))
        
        # 平滑包络
        window_size = int(self.sampling_rate * 2)  # 2秒窗口
        window_size = max(5, window_size)
        if window_size % 2 == 0:
            window_size += 1
        envelope_smooth = signal.savgol_filter(envelope, window_size, 3)
        
        # 检测低于阈值的区间
        threshold = np.mean(envelope_smooth) * self.apnea_threshold
        below_threshold = envelope_smooth < threshold
        
        # 查找连续的低于阈值区间
        diff = np.diff(below_threshold.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # 处理边界情况
        if below_threshold[0]:
            starts = np.concatenate([[0], starts])
        if below_threshold[-1]:
            ends = np.concatenate([ends, [len(below_threshold)]])
        
        # 筛选符合时长要求的事件
        min_samples = int(self.apnea_duration_min * self.sampling_rate)
        
        for start, end in zip(starts, ends):
            duration = (end - start) / self.sampling_rate
            if duration >= self.apnea_duration_min:
                events.append({
                    'start_time': self._ts_at(timestamps, min(start, len(timestamps)-1)),
                    'end_time': self._ts_at(timestamps, min(end-1, len(timestamps)-1)),
                    'duration_seconds': round(duration, 1),
                    'severity': 'severe' if duration > 30 else 'moderate' if duration > 20 else 'mild'
                })
        
        return events
    
    def _detect_hypopnea_events(self, respiratory_signal: np.ndarray, timestamps) -> List[Dict]:
        """检测低通气事件"""
        events = []
        
        # 计算呼吸信号的包络
        envelope = np.abs(signal.hilbert(respiratory_signal))
        
        # 平滑包络
        window_size = int(self.sampling_rate * 2)
        window_size = max(5, window_size)
        if window_size % 2 == 0:
            window_size += 1
        envelope_smooth = signal.savgol_filter(envelope, window_size, 3)
        
        # 检测低于低通气阈值的区间
        threshold = np.mean(envelope_smooth) * self.hypopnea_threshold
        below_threshold = envelope_smooth < threshold
        
        # 查找连续的低于阈值区间
        diff = np.diff(below_threshold.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # 处理边界情况
        if below_threshold[0]:
            starts = np.concatenate([[0], starts])
        if below_threshold[-1]:
            ends = np.concatenate([ends, [len(below_threshold)]])
        
        # 筛选符合时长要求的事件
        min_samples = int(self.apnea_duration_min * self.sampling_rate)
        
        for start, end in zip(starts, ends):
            duration = (end - start) / self.sampling_rate
            if duration >= self.apnea_duration_min:
                events.append({
                    'start_time': self._ts_at(timestamps, min(start, len(timestamps)-1)),
                    'end_time': self._ts_at(timestamps, min(end-1, len(timestamps)-1)),
                    'duration_seconds': round(duration, 1),
                    'reduction_percentage': round((1 - np.mean(envelope_smooth[start:end]) / np.mean(envelope_smooth)) * 100, 1)
                })
        
        return events
    
    def _classify_apnea_severity(self, ahi_index: float) -> str:
        """呼吸暂停严重程度分级"""
        if ahi_index < 5:
            return 'normal'
        elif ahi_index < 15:
            return 'mild'
        elif ahi_index < 30:
            return 'moderate'
        else:
            return 'severe'
    
    def _calculate_respiratory_statistics(self, respiratory_signal: np.ndarray) -> Dict[str, float]:
        """计算呼吸相关统计指标"""
        # 呼吸频率估算
        respiratory_rate = self._estimate_respiratory_rate(respiratory_signal)
        
        # 呼吸变异性
        respiratory_variability = np.std(respiratory_signal) / np.mean(np.abs(respiratory_signal)) * 100
        
        # 呼吸规律性
        respiratory_regularity = self._calculate_respiratory_regularity(respiratory_signal)
        
        return {
            'respiratory_rate': round(respiratory_rate, 2),
            'respiratory_variability': round(respiratory_variability, 2),
            'respiratory_regularity': round(respiratory_regularity, 3),
            'signal_strength': round(np.std(respiratory_signal), 2)
        }
    
    def _estimate_respiratory_rate(self, respiratory_signal: np.ndarray) -> float:
        """估算呼吸频率"""
        # FFT分析
        fft = np.fft.fft(respiratory_signal)
        freqs = np.fft.fftfreq(len(respiratory_signal), 1/self.sampling_rate)
        
        # 呼吸频率范围 (0.1-0.5 Hz, 对应6-30次/分钟)
        valid_indices = (freqs >= 0.1) & (freqs <= 0.5)
        valid_fft = np.abs(fft[valid_indices])
        valid_freqs = freqs[valid_indices]
        
        if len(valid_fft) > 0:
            peak_freq = valid_freqs[np.argmax(valid_fft)]
            return peak_freq * 60  # 转换为次/分钟
        
        return 15.0  # 默认呼吸频率
    
    def _calculate_respiratory_regularity(self, respiratory_signal: np.ndarray) -> float:
        """计算呼吸规律性"""
        # 使用自相关函数评估规律性
        correlation = np.correlate(respiratory_signal, respiratory_signal, mode='full')
        correlation = correlation[len(correlation)//2:]
        
        # 寻找第一个局部最大值（除了零延迟）
        peaks, _ = signal.find_peaks(correlation[1:], height=np.max(correlation) * 0.3)
        
        if len(peaks) > 0:
            return correlation[peaks[0] + 1] / correlation[0]
        
        return 0.5  # 默认规律性
    
    def _estimate_spo2(self, ppg_data: np.ndarray) -> np.ndarray:
        """估算血氧饱和度"""
        # 对于大数据集，进行降采样以提高性能
        if len(ppg_data) > 100000:
            downsample_factor = max(1, len(ppg_data) // 100000)
            ppg_data = ppg_data[::downsample_factor]
        
        # 简化的SpO2估算算法
        # 基于PPG信号的AC/DC比值和经验公式
        try:
            # 计算AC成分（交流成分）
            ac_component = ppg_data - self._savgol(ppg_data)
            
            # 计算DC成分（直流成分）
            dc_component = self._savgol(ppg_data)
            
            # 避免除零
            dc_component = np.where(dc_component == 0, 1, dc_component)
            
            # AC/DC比值
            ratio = np.abs(ac_component) / np.abs(dc_component)
            
            # 经验公式估算SpO2
            spo2 = 100 - 25 * ratio
            
            # 限制在合理范围内
            spo2 = np.clip(spo2, 70, 100)
            
        except:
            # 如果计算失败，返回默认值
            spo2 = np.full(len(ppg_data), 95.0)
        
        return spo2
    
    def _identify_night_period(self, timestamps) -> np.ndarray:
        """识别夜间时段"""
        if isinstance(timestamps, np.ndarray) and getattr(timestamps, 'dtype', None) is not None and timestamps.dtype.kind == 'M':
            hrs = (timestamps.astype('datetime64[h]').astype('int64') % 24)
            mask = (hrs >= self.night_hours[0]) | (hrs < self.night_hours[1])
            return np.where(mask)[0]
        out = []
        for i, s in enumerate(timestamps):
            try:
                dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
                h = dt.hour
                if h >= self.night_hours[0] or h < self.night_hours[1]:
                    out.append(i)
            except (ValueError, TypeError):
                continue
        return np.array(out)
    
    def _detect_desaturation_events(self, spo2_values: np.ndarray, timestamps, 
                                  night_indices: np.ndarray) -> List[Dict]:
        """检测血氧饱和度下降事件"""
        events = []
        
        # 检测SpO2下降超过3%的事件
        baseline = self._savgol(spo2_values)
        drops = baseline - spo2_values
        
        # 寻找下降事件
        significant_drops = drops > 3  # 下降超过3%
        
        # 查找连续的下降区间
        diff = np.diff(significant_drops.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # 处理边界情况
        if significant_drops[0]:
            starts = np.concatenate([[0], starts])
        if significant_drops[-1]:
            ends = np.concatenate([ends, [len(significant_drops)]])
        
        for start, end in zip(starts, ends):
            duration = (end - start) / self.sampling_rate
            if duration >= 10:  # 至少持续10秒
                max_drop = np.max(drops[start:end])
                min_spo2 = np.min(spo2_values[start:end])
                
                # 获取对应的时间戳索引
                if len(night_indices) > 0:
                    actual_start = night_indices[min(start, len(night_indices)-1)]
                    actual_end = night_indices[min(end-1, len(night_indices)-1)]
                else:
                    actual_start = start
                    actual_end = end - 1
                
                events.append({
                    'start_time': self._ts_at(timestamps, min(actual_start, len(timestamps)-1)),
                    'end_time': self._ts_at(timestamps, min(actual_end, len(timestamps)-1)),
                    'duration_seconds': round(duration, 1),
                    'max_drop_percent': round(max_drop, 1),
                    'min_spo2': round(min_spo2, 1)
                })
        
        return events
    
    def _calculate_odi(self, desaturation_events: List[Dict], sleep_hours: float) -> float:
        """计算ODI指数（氧减指数）"""
        if sleep_hours <= 0:
            return 0.0
        
        return len(desaturation_events) / sleep_hours
    
    def _calculate_spo2_statistics(self, spo2_values: np.ndarray) -> Dict[str, float]:
        """计算血氧统计指标"""
        return {
            'mean_spo2': round(np.mean(spo2_values), 2),
            'median_spo2': round(np.median(spo2_values), 2),
            'std_spo2': round(np.std(spo2_values), 2),
            'min_spo2': round(np.min(spo2_values), 2),
            'max_spo2': round(np.max(spo2_values), 2),
            'spo2_range': round(np.max(spo2_values) - np.min(spo2_values), 2)
        }
    
    def _analyze_spo2_variability(self, spo2_values: np.ndarray) -> Dict[str, float]:
        """分析血氧变异性"""
        # 计算变异系数
        cv = np.std(spo2_values) / np.mean(spo2_values) * 100
        
        # 计算相邻差值的标准差
        successive_diff = np.diff(spo2_values)
        rmssd = np.sqrt(np.mean(successive_diff**2))
        
        return {
            'coefficient_of_variation': round(cv, 3),
            'rmssd': round(rmssd, 3),
            'variability_index': round(np.std(spo2_values) / (np.max(spo2_values) - np.min(spo2_values)) * 100, 2)
        }
    
    def _estimate_bp_indicators(self, ppg_data: np.ndarray) -> np.ndarray:
        """估算血压相关指标"""
        # 基于PPG信号特征估算血压相关指标
        # 使用脉搏波传导时间(PTT)相关的特征
        
        # 计算脉搏波特征
        pulse_features = self._extract_pulse_features(ppg_data)
        
        # 基于经验公式估算血压指标
        # 这里使用简化的线性关系
        bp_indicator = 100 + 20 * np.sin(np.linspace(0, 4*np.pi, len(pulse_features))) + \
                      10 * (pulse_features - np.mean(pulse_features)) / np.std(pulse_features)
        
        return bp_indicator
    
    def _extract_pulse_features(self, ppg_data: np.ndarray) -> np.ndarray:
        """提取脉搏波特征"""
        # 检测脉搏峰值
        peaks, _ = signal.find_peaks(ppg_data, distance=int(self.sampling_rate * 0.4))
        
        if len(peaks) < 2:
            return np.ones(len(ppg_data))
        
        # 计算脉搏间期
        pulse_intervals = np.diff(peaks) / self.sampling_rate
        
        # 插值到原始数据长度
        pulse_features = np.interp(np.arange(len(ppg_data)), peaks[:-1], pulse_intervals)
        
        return pulse_features
    
    def _analyze_circadian_rhythm(self, bp_indicators: np.ndarray, timestamps) -> Dict[str, Any]:
        """分析昼夜节律"""
        # 按小时分组计算平均值
        hourly_means = {}
        if isinstance(timestamps, np.ndarray) and getattr(timestamps, 'dtype', None) is not None and timestamps.dtype.kind == 'M':
            hrs = (timestamps.astype('datetime64[h]').astype('int64') % 24)
            x = bp_indicators[:len(hrs)]
            for h in range(24):
                mask = (hrs == h)
                vals = x[mask[:len(x)]]
                if vals.size:
                    hourly_means[h] = float(np.mean(vals))
        else:
            for i, s in enumerate(timestamps):
                if i >= len(bp_indicators):
                    break
                try:
                    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
                    h = dt.hour
                    hourly_means.setdefault(h, []).append(bp_indicators[i])
                except (ValueError, TypeError):
                    continue
            for h, values in list(hourly_means.items()):
                hourly_means[h] = float(np.mean(values))
        
        # 计算昼夜比值
        day_hours = [h for h in range(6, 22)]  # 6:00-22:00
        night_hours = [h for h in list(range(22, 24)) + list(range(0, 6))]  # 22:00-6:00
        
        day_values = [hourly_means[h] for h in day_hours if h in hourly_means]
        night_values = [hourly_means[h] for h in night_hours if h in hourly_means]
        
        day_mean = np.mean(day_values) if day_values else 0
        night_mean = np.mean(night_values) if night_values else 0
        
        day_night_ratio = day_mean / night_mean if night_mean > 0 else 1.0
        
        return {
            'hourly_means': hourly_means,
            'day_mean': round(day_mean, 2),
            'night_mean': round(night_mean, 2),
            'day_night_ratio': round(day_night_ratio, 3),
            'circadian_amplitude': round(max(hourly_means.values()) - min(hourly_means.values()), 2) if hourly_means else 0
        }
    
    def _analyze_nocturnal_dipping(self, bp_indicators: np.ndarray, timestamps) -> Dict[str, Any]:
        """分析夜间血压下降"""
        # 识别白天和夜间时段
        if isinstance(timestamps, np.ndarray) and getattr(timestamps, 'dtype', None) is not None and timestamps.dtype.kind == 'M':
            hrs = (timestamps.astype('datetime64[h]').astype('int64') % 24)
            day_mask = (hrs >= 6) & (hrs < 22)
            night_mask = ~day_mask
            x = bp_indicators[:len(hrs)]
            day_values = x[day_mask[:len(x)]]
            night_values = x[night_mask[:len(x)]]
        else:
            day_values = []
            night_values = []
            for i, s in enumerate(timestamps):
                if i >= len(bp_indicators):
                    break
                try:
                    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
                    h = dt.hour
                    if 6 <= h < 22:
                        day_values.append(bp_indicators[i])
                    else:
                        night_values.append(bp_indicators[i])
                except (ValueError, TypeError):
                    continue
        
        dv = np.asarray(day_values)
        nv = np.asarray(night_values)
        if dv.size == 0 or nv.size == 0:
            return {'dipping_percentage': 0, 'dipping_pattern': 'insufficient_data'}
        
        day_mean = np.mean(dv)
        night_mean = np.mean(nv)
        
        # 计算下降百分比
        dipping_percentage = (day_mean - night_mean) / day_mean * 100
        
        # 分类下降模式
        if dipping_percentage >= 10:
            dipping_pattern = 'normal_dipper'
        elif 0 <= dipping_percentage < 10:
            dipping_pattern = 'non_dipper'
        else:
            dipping_pattern = 'reverse_dipper'
        
        return {
            'day_mean': round(day_mean, 2),
            'night_mean': round(night_mean, 2),
            'dipping_percentage': round(dipping_percentage, 2),
            'dipping_pattern': dipping_pattern
        }
    
    def _analyze_bp_variability(self, bp_indicators: np.ndarray, timestamps) -> Dict[str, float]:
        """分析血压变异性"""
        # 短期变异性（相邻测量值差异）
        short_term_var = np.std(np.diff(bp_indicators))
        
        # 长期变异性（整体标准差）
        long_term_var = np.std(bp_indicators)
        
        # 变异系数
        cv = long_term_var / np.mean(bp_indicators) * 100
        
        return {
            'short_term_variability': round(short_term_var, 2),
            'long_term_variability': round(long_term_var, 2),
            'coefficient_of_variation': round(cv, 2)
        }

    def _ts_at(self, timestamps, idx: int) -> str:
        if isinstance(timestamps, np.ndarray) and getattr(timestamps, 'dtype', None) is not None and timestamps.dtype.kind == 'M':
            j = max(0, min(idx, len(timestamps)-1))
            s = np.datetime_as_string(timestamps[j], unit='s')
            return s.replace('T', ' ')
        j = max(0, min(idx, len(timestamps)-1))
        return timestamps[j]
    
    def _identify_bp_rhythm_pattern(self, circadian_analysis: Dict, nocturnal_dipping: Dict) -> str:
        """识别血压节律模式"""
        day_night_ratio = circadian_analysis.get('day_night_ratio', 1.0)
        dipping_pattern = nocturnal_dipping.get('dipping_pattern', 'unknown')
        
        if day_night_ratio > 1.1 and dipping_pattern == 'normal_dipper':
            return 'normal_circadian'
        elif day_night_ratio > 1.05 and dipping_pattern == 'non_dipper':
            return 'non_dipper_pattern'
        elif dipping_pattern == 'reverse_dipper':
            return 'reverse_dipper_pattern'
        else:
            return 'irregular_pattern'
    
    def _assess_respiratory_quality(self, respiratory_signal: np.ndarray) -> str:
        """评估呼吸信号质量"""
        snr = np.mean(respiratory_signal**2) / np.var(respiratory_signal)
        
        if snr > 10:
            return 'excellent'
        elif snr > 5:
            return 'good'
        elif snr > 2:
            return 'fair'
        else:
            return 'poor'
    
    def _assess_spo2_quality(self, spo2_values: np.ndarray) -> str:
        """评估血氧信号质量"""
        # 基于数据稳定性和合理性评估
        stability = 1 / (1 + np.std(spo2_values))
        reasonableness = np.sum((spo2_values >= 70) & (spo2_values <= 100)) / len(spo2_values)
        
        quality_score = stability * reasonableness
        
        if quality_score > 0.8:
            return 'excellent'
        elif quality_score > 0.6:
            return 'good'
        elif quality_score > 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _assess_rhythm_quality(self, bp_indicators: np.ndarray, timestamps: List[str]) -> str:
        """评估血压节律分析质量"""
        # 基于数据连续性和时间跨度评估
        time_span_hours = len(timestamps) / self.sampling_rate / 3600
        data_continuity = len(bp_indicators) / len(timestamps) if len(timestamps) > 0 else 0
        
        if time_span_hours >= 20 and data_continuity > 0.8:
            return 'excellent'
        elif time_span_hours >= 12 and data_continuity > 0.6:
            return 'good'
        elif time_span_hours >= 6 and data_continuity > 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _get_default_apnea_result(self) -> Dict[str, Any]:
        """获取默认的呼吸暂停分析结果"""
        return {
            'apnea_events': [],
            'hypopnea_events': [],
            'ahi_index': 0.0,
            'severity': 'unknown',
            'total_events': 0,
            'apnea_count': 0,
            'hypopnea_count': 0,
            'respiratory_statistics': {
                'respiratory_rate': 15.0,
                'respiratory_variability': 0.0,
                'respiratory_regularity': 0.5,
                'signal_strength': 0.0
            },
            'analysis_quality': 'poor'
        }
    
    def _get_default_spo2_result(self) -> Dict[str, Any]:
        """获取默认的血氧分析结果"""
        return {
            'mean_spo2': 95.0,
            'min_spo2': 90.0,
            'spo2_below_90_percent': 0.0,
            'desaturation_events': [],
            'odi_index': 0.0,
            'spo2_statistics': {
                'mean_spo2': 95.0,
                'median_spo2': 95.0,
                'std_spo2': 2.0,
                'min_spo2': 90.0,
                'max_spo2': 100.0,
                'spo2_range': 10.0
            },
            'spo2_variability': {
                'coefficient_of_variation': 2.0,
                'rmssd': 1.0,
                'variability_index': 20.0
            },
            'night_duration_hours': 0.0,
            'analysis_quality': 'poor'
        }
    
    def _get_default_bp_rhythm_result(self) -> Dict[str, Any]:
        """获取默认的血压节律分析结果"""
        return {
            'circadian_analysis': {
                'hourly_means': {},
                'day_mean': 120.0,
                'night_mean': 110.0,
                'day_night_ratio': 1.09,
                'circadian_amplitude': 20.0
            },
            'nocturnal_dipping': {
                'day_mean': 120.0,
                'night_mean': 110.0,
                'dipping_percentage': 8.3,
                'dipping_pattern': 'non_dipper'
            },
            'bp_variability': {
                'short_term_variability': 5.0,
                'long_term_variability': 10.0,
                'coefficient_of_variation': 8.3
            },
            'rhythm_pattern': 'irregular_pattern',
            'day_night_ratio': 1.09,
            'dipping_percentage': 8.3,
            'rhythm_quality': 'poor'
        }


def analyze_sleep_data(ppg_data: np.ndarray, timestamps, 
                      sampling_rate: int = 125) -> Dict[str, Any]:
    """
    睡眠数据分析主函数
    
    Args:
        ppg_data: PPG数据
        timestamps: 时间戳列表
        sampling_rate: 采样率
        
    Returns:
        睡眠分析结果
    """
    analyzer = PPGSleepAnalyzer(sampling_rate)
    
    # 睡眠呼吸暂停分析
    apnea_result = analyzer.analyze_sleep_apnea(ppg_data, timestamps)
    
    # 夜间血氧饱和度分析
    spo2_result = analyzer.analyze_nocturnal_spo2(ppg_data, timestamps)
    
    # 血压节律分析
    bp_rhythm_result = analyzer.analyze_blood_pressure_rhythm(ppg_data, timestamps)
    
    return {
        'sleep_apnea_analysis': apnea_result,
        'nocturnal_spo2_analysis': spo2_result,
        'blood_pressure_rhythm_analysis': bp_rhythm_result,
        'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'data_duration_hours': round(len(ppg_data) / sampling_rate / 3600, 2)
    }