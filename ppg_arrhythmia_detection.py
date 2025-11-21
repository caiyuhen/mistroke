#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPG心律不齐检测模块
基于PPG信号进行心律不齐检测，包括房颤、早搏等异常心律的识别
"""

import numpy as np
import scipy.signal as signal
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class PPGArrhythmiaDetector:
    def __init__(self, sampling_rate=125):
        """
        初始化PPG心律不齐检测器
        
        Args:
            sampling_rate: 采样率，默认125Hz
        """
        self.sampling_rate = sampling_rate
        self.min_peak_distance = int(0.4 * sampling_rate)  # 最小峰值间隔400ms
        self.max_peak_distance = int(2.0 * sampling_rate)  # 最大峰值间隔2s
        
        # 滤波器参数
        self.lowcut = 0.5   # 低频截止
        self.highcut = 8.0  # 高频截止
        
        # 心律不齐检测阈值
        self.cv_threshold = 15.0  # 变异系数阈值(%)
        self.rmssd_threshold = 50.0  # RMSSD阈值(ms)
        self.pnn50_threshold = 10.0  # pNN50阈值(%)
        
    def preprocess_signal(self, ppg_signal):
        """
        PPG信号预处理
        
        Args:
            ppg_signal: 原始PPG信号
            
        Returns:
            processed_signal: 预处理后的信号
            signal_quality: 信号质量评分
        """
        try:
            # 转换为numpy数组
            signal_array = np.array(ppg_signal, dtype=float)
            
            # 去除异常值
            signal_array = self._remove_outliers(signal_array)
            
            # 带通滤波
            filtered_signal = self._bandpass_filter(signal_array)
            
            # 信号质量评估
            sqi = self._calculate_signal_quality(filtered_signal)
            
            # 归一化
            normalized_signal = self._normalize_signal(filtered_signal)
            
            return normalized_signal, sqi
            
        except Exception as e:
            print(f"信号预处理失败: {e}")
            return None, 0.0
    
    def _remove_outliers(self, signal_array):
        """去除异常值"""
        q75, q25 = np.percentile(signal_array, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        # 将异常值替换为边界值
        signal_array = np.clip(signal_array, lower_bound, upper_bound)
        return signal_array
    
    def _bandpass_filter(self, signal_array):
        """带通滤波器"""
        try:
            nyquist = 0.5 * self.sampling_rate
            low = self.lowcut / nyquist
            high = self.highcut / nyquist
            
            # 设计Butterworth滤波器
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_signal = signal.filtfilt(b, a, signal_array)
            
            return filtered_signal
        except:
            # 如果滤波失败，返回原信号
            return signal_array
    
    def _calculate_signal_quality(self, signal_array):
        try:
            signal_power = np.var(signal_array)
            noise_estimate = np.var(np.diff(signal_array))
            snr = 10 * np.log10(signal_power / (noise_estimate + 1e-10))

            baseline_stability = 1.0 - (np.std(signal_array) / (np.mean(np.abs(signal_array)) + 1e-10))

            use_fft = os.environ.get('FAST_AUTOCORR', '0') == '1'
            if use_fft:
                n = signal_array.size
                nfft = 1 << (n - 1).bit_length()
                X = np.fft.rfft(signal_array, nfft)
                acf = np.fft.irfft(X * np.conjugate(X), nfft)
                acf = acf[:n]
            else:
                acf = np.correlate(signal_array, signal_array, mode='full')
                acf = acf[acf.size // 2:]

            start = self.min_peak_distance
            end = min(self.max_peak_distance, acf.size)
            periodicity = np.max(acf[start:end]) / (acf[0] + 1e-12) if end > start and acf.size > 0 else 0.0

            sqi = (np.clip(snr / 20, 0, 1) * 0.4 + np.clip(baseline_stability, 0, 1) * 0.3 + np.clip(periodicity, 0, 1) * 0.3)

            return float(sqi)
        except:
            return 0.5
    
    def _normalize_signal(self, signal_array):
        """信号归一化"""
        signal_min = np.min(signal_array)
        signal_max = np.max(signal_array)
        
        if signal_max - signal_min > 0:
            normalized = (signal_array - signal_min) / (signal_max - signal_min)
        else:
            normalized = signal_array
            
        return normalized
    
    def detect_peaks(self, ppg_signal):
        """
        检测PPG信号中的脉搏波峰值
        
        Args:
            ppg_signal: 预处理后的PPG信号
            
        Returns:
            peaks: 峰值位置数组
            peak_properties: 峰值属性
        """
        try:
            # 自适应阈值
            signal_std = np.std(ppg_signal)
            signal_mean = np.mean(ppg_signal)
            height_threshold = signal_mean + 0.3 * signal_std
            
            # 寻找峰值
            peaks, properties = signal.find_peaks(
                ppg_signal,
                height=height_threshold,
                distance=self.min_peak_distance,
                prominence=0.1 * signal_std
            )
            
            # 形态学验证
            validated_peaks = self._validate_peaks(ppg_signal, peaks)
            
            return validated_peaks, properties
            
        except Exception as e:
            print(f"峰值检测失败: {e}")
            return np.array([]), {}
    
    def _validate_peaks(self, signal_array, peaks):
        """验证峰值的形态学特征"""
        validated_peaks = []
        
        for peak in peaks:
            # 检查峰值周围的形态
            start_idx = max(0, peak - self.min_peak_distance // 4)
            end_idx = min(len(signal_array), peak + self.min_peak_distance // 4)
            
            if end_idx - start_idx < self.min_peak_distance // 2:
                continue
                
            # 验证是否为局部最大值
            local_signal = signal_array[start_idx:end_idx]
            local_peak_idx = peak - start_idx
            
            if (local_peak_idx > 0 and local_peak_idx < len(local_signal) - 1 and
                signal_array[peak] > signal_array[peak - 1] and
                signal_array[peak] > signal_array[peak + 1]):
                validated_peaks.append(peak)
        
        return np.array(validated_peaks)
    
    def extract_ppi_sequence(self, peaks):
        """
        提取Peak-to-Peak间隔(PPI)序列
        
        Args:
            peaks: 峰值位置数组
            
        Returns:
            ppi_sequence: PPI序列(秒)
            ppi_ms: PPI序列(毫秒)
        """
        if len(peaks) < 2:
            return np.array([]), np.array([])
        
        # 计算相邻峰值间隔
        ppi_samples = np.diff(peaks)
        ppi_seconds = ppi_samples / self.sampling_rate
        ppi_ms = ppi_seconds * 1000
        
        # 过滤异常间隔
        valid_ppi = self._filter_abnormal_ppi(ppi_ms)
        
        return ppi_seconds[valid_ppi], ppi_ms[valid_ppi]
    
    def _filter_abnormal_ppi(self, ppi_ms):
        """过滤异常的PPI值"""
        # 生理范围：300ms - 2000ms
        min_ppi = 300
        max_ppi = 2000
        
        valid_mask = (ppi_ms >= min_ppi) & (ppi_ms <= max_ppi)
        
        # 进一步过滤极端异常值
        if np.sum(valid_mask) > 0:
            median_ppi = np.median(ppi_ms[valid_mask])
            mad = np.median(np.abs(ppi_ms[valid_mask] - median_ppi))
            
            # 使用修正的Z分数
            modified_z_scores = 0.6745 * (ppi_ms - median_ppi) / (mad + 1e-10)
            outlier_mask = np.abs(modified_z_scores) < 3.5
            
            valid_mask = valid_mask & outlier_mask
        
        return valid_mask
    
    def calculate_time_domain_features(self, ppi_ms):
        """
        计算时域特征参数
        
        Args:
            ppi_ms: PPI序列(毫秒)
            
        Returns:
            features: 时域特征字典
        """
        if len(ppi_ms) < 2:
            return self._empty_time_features()
        
        try:
            # 基本统计量
            mean_ppi = np.mean(ppi_ms)
            std_ppi = np.std(ppi_ms, ddof=1)
            
            # SDNN: PPI序列的标准差
            sdnn = std_ppi
            
            # RMSSD: 相邻PPI差值的均方根
            ppi_diff = np.diff(ppi_ms)
            rmssd = np.sqrt(np.mean(ppi_diff ** 2)) if len(ppi_diff) > 0 else 0
            
            # pNN50: 相邻PPI差值>50ms的百分比
            pnn50 = (np.sum(np.abs(ppi_diff) > 50) / len(ppi_diff) * 100) if len(ppi_diff) > 0 else 0
            
            # 变异系数
            cv = (std_ppi / mean_ppi * 100) if mean_ppi > 0 else 0
            
            # 不规律指数
            irregularity_indices = []
            for i in range(len(ppi_ms) - 1):
                if ppi_ms[i] > 0:
                    irregularity = abs(ppi_ms[i+1] - ppi_ms[i]) / ppi_ms[i] * 100
                    irregularity_indices.append(irregularity)
            
            mean_irregularity = np.mean(irregularity_indices) if irregularity_indices else 0
            
            # 三角指数
            triangular_index = len(ppi_ms) / (2 * np.max(np.histogram(ppi_ms, bins=50)[0])) if len(ppi_ms) > 0 else 0
            
            features = {
                'mean_ppi': float(mean_ppi),
                'sdnn': float(sdnn),
                'rmssd': float(rmssd),
                'pnn50': float(pnn50),
                'cv': float(cv),
                'mean_irregularity': float(mean_irregularity),
                'triangular_index': float(triangular_index),
                'min_ppi': float(np.min(ppi_ms)),
                'max_ppi': float(np.max(ppi_ms)),
                'range_ppi': float(np.max(ppi_ms) - np.min(ppi_ms))
            }
            
            return features
            
        except Exception as e:
            print(f"时域特征计算失败: {e}")
            return self._empty_time_features()
    
    def _empty_time_features(self):
        """返回空的时域特征"""
        return {
            'mean_ppi': 0.0, 'sdnn': 0.0, 'rmssd': 0.0, 'pnn50': 0.0,
            'cv': 0.0, 'mean_irregularity': 0.0, 'triangular_index': 0.0,
            'min_ppi': 0.0, 'max_ppi': 0.0, 'range_ppi': 0.0
        }
    
    def calculate_frequency_domain_features(self, ppi_seconds):
        """
        计算频域特征参数
        
        Args:
            ppi_seconds: PPI序列(秒)
            
        Returns:
            features: 频域特征字典
        """
        if len(ppi_seconds) < 10:
            return self._empty_freq_features()
        
        try:
            # 重采样到均匀时间间隔
            time_points = np.cumsum(ppi_seconds)
            time_points = np.insert(time_points, 0, 0)
            
            # 插值到4Hz采样率
            fs_resample = 4.0
            time_uniform = np.arange(0, time_points[-1], 1/fs_resample)
            
            if len(time_uniform) < 10:
                return self._empty_freq_features()
            
            ppi_interp = np.interp(time_uniform, time_points[:-1], ppi_seconds)
            
            # 去趋势
            ppi_detrend = signal.detrend(ppi_interp)
            
            # 计算功率谱密度
            freqs, psd = signal.welch(ppi_detrend, fs=fs_resample, nperseg=min(256, len(ppi_detrend)))
            
            # 频段划分
            vlf_band = (freqs >= 0.003) & (freqs < 0.04)
            lf_band = (freqs >= 0.04) & (freqs < 0.15)
            hf_band = (freqs >= 0.15) & (freqs < 0.4)
            
            # 计算各频段功率
            vlf_power = np.trapz(psd[vlf_band], freqs[vlf_band]) if np.any(vlf_band) else 0
            lf_power = np.trapz(psd[lf_band], freqs[lf_band]) if np.any(lf_band) else 0
            hf_power = np.trapz(psd[hf_band], freqs[hf_band]) if np.any(hf_band) else 0
            
            total_power = vlf_power + lf_power + hf_power
            
            # 归一化功率
            lf_norm = (lf_power / (lf_power + hf_power) * 100) if (lf_power + hf_power) > 0 else 0
            hf_norm = (hf_power / (lf_power + hf_power) * 100) if (lf_power + hf_power) > 0 else 0
            
            # LF/HF比值
            lf_hf_ratio = (lf_power / hf_power) if hf_power > 0 else 0
            
            features = {
                'vlf_power': float(vlf_power),
                'lf_power': float(lf_power),
                'hf_power': float(hf_power),
                'total_power': float(total_power),
                'lf_norm': float(lf_norm),
                'hf_norm': float(hf_norm),
                'lf_hf_ratio': float(lf_hf_ratio)
            }
            
            return features
            
        except Exception as e:
            print(f"频域特征计算失败: {e}")
            return self._empty_freq_features()
    
    def _empty_freq_features(self):
        """返回空的频域特征"""
        return {
            'vlf_power': 0.0, 'lf_power': 0.0, 'hf_power': 0.0,
            'total_power': 0.0, 'lf_norm': 0.0, 'hf_norm': 0.0,
            'lf_hf_ratio': 0.0
        }
    
    def detect_atrial_fibrillation(self, ppi_ms, time_features):
        """
        房颤检测算法
        
        Args:
            ppi_ms: PPI序列(毫秒)
            time_features: 时域特征
            
        Returns:
            afib_result: 房颤检测结果
        """
        if len(ppi_ms) < 10:
            return {
                'afib_detected': False,
                'confidence': 0.0,
                'risk_level': '无法评估'
            }
        
        try:
            # 获取关键指标
            cv = time_features.get('cv', 0)
            rmssd = time_features.get('rmssd', 0)
            mean_irregularity = time_features.get('mean_irregularity', 0)
            
            # 房颤检测规则
            afib_score = 0
            
            # 规则1: 变异系数>15%
            if cv > self.cv_threshold:
                afib_score += 3
            elif cv > 10:
                afib_score += 1
            
            # 规则2: RMSSD>50ms
            if rmssd > self.rmssd_threshold:
                afib_score += 2
            elif rmssd > 30:
                afib_score += 1
            
            # 规则3: 不规律性>30%
            if mean_irregularity > 30:
                afib_score += 2
            elif mean_irregularity > 20:
                afib_score += 1
            
            # 规则4: 连续异常间隔检测
            consecutive_irregular = self._count_consecutive_irregular_intervals(ppi_ms)
            if consecutive_irregular > len(ppi_ms) * 0.3:
                afib_score += 2
            
            # 判断结果
            if afib_score >= 6:
                afib_detected = True
                risk_level = '高度疑似房颤'
                confidence = min(0.9, afib_score / 10)
            elif afib_score >= 4:
                afib_detected = True
                risk_level = '疑似房颤'
                confidence = min(0.7, afib_score / 10)
            elif afib_score >= 2:
                afib_detected = False
                risk_level = '心律轻度不规律'
                confidence = min(0.5, afib_score / 10)
            else:
                afib_detected = False
                risk_level = '心律正常'
                confidence = 0.1
            
            return {
                'afib_detected': afib_detected,
                'confidence': float(confidence),
                'risk_level': risk_level,
                'afib_score': afib_score,
                'cv': cv,
                'rmssd': rmssd,
                'mean_irregularity': mean_irregularity
            }
            
        except Exception as e:
            print(f"房颤检测失败: {e}")
            return {
                'afib_detected': False,
                'confidence': 0.0,
                'risk_level': '检测失败'
            }
    
    def _count_consecutive_irregular_intervals(self, ppi_ms):
        """计算连续不规律间隔数量"""
        if len(ppi_ms) < 2:
            return 0
        
        irregular_count = 0
        for i in range(len(ppi_ms) - 1):
            if ppi_ms[i] > 0:
                change_rate = abs(ppi_ms[i+1] - ppi_ms[i]) / ppi_ms[i]
                if change_rate > 0.2:  # 变化率>20%
                    irregular_count += 1
        
        return irregular_count
    
    def detect_premature_beats(self, ppi_ms):
        """
        早搏检测(房性早搏PAC和室性早搏PVC)
        
        Args:
            ppi_ms: PPI序列(毫秒)
            
        Returns:
            premature_beats: 早搏检测结果
        """
        if len(ppi_ms) < 3:
            return {
                'pac_count': 0,
                'pvc_count': 0,
                'total_premature': 0,
                'premature_rate': 0.0
            }
        
        try:
            pac_count = 0
            pvc_count = 0
            
            # 计算基线PPI
            median_ppi = np.median(ppi_ms)
            
            for i in range(1, len(ppi_ms) - 1):
                current_ppi = ppi_ms[i]
                next_ppi = ppi_ms[i + 1]
                
                # 检测短间隔(早搏)
                if current_ppi < 0.8 * median_ppi:
                    # 检查代偿间歇
                    if next_ppi > 1.2 * median_ppi:
                        # 完全代偿间歇 -> 可能是PVC
                        if next_ppi > 1.5 * median_ppi:
                            pvc_count += 1
                        else:
                            # 不完全代偿间歇 -> 可能是PAC
                            pac_count += 1
            
            total_premature = pac_count + pvc_count
            premature_rate = (total_premature / len(ppi_ms) * 100) if len(ppi_ms) > 0 else 0
            
            return {
                'pac_count': pac_count,
                'pvc_count': pvc_count,
                'total_premature': total_premature,
                'premature_rate': float(premature_rate)
            }
            
        except Exception as e:
            print(f"早搏检测失败: {e}")
            return {
                'pac_count': 0,
                'pvc_count': 0,
                'total_premature': 0,
                'premature_rate': 0.0
            }
    
    def detect_sinus_arrhythmia(self, freq_features):
        """
        窦性心律不齐检测
        
        Args:
            freq_features: 频域特征
            
        Returns:
            sinus_arrhythmia: 窦性心律不齐检测结果
        """
        try:
            hf_power = freq_features.get('hf_power', 0)
            total_power = freq_features.get('total_power', 0)
            lf_hf_ratio = freq_features.get('lf_hf_ratio', 0)
            
            # 窦性心律不齐特征：HF成分显著增加
            hf_ratio = (hf_power / total_power) if total_power > 0 else 0
            
            if hf_ratio > 0.4 and lf_hf_ratio < 1.0:
                sinus_arrhythmia_detected = True
                severity = '显著' if hf_ratio > 0.6 else '轻度'
            else:
                sinus_arrhythmia_detected = False
                severity = '无'
            
            return {
                'sinus_arrhythmia_detected': sinus_arrhythmia_detected,
                'severity': severity,
                'hf_ratio': float(hf_ratio),
                'lf_hf_ratio': float(lf_hf_ratio)
            }
            
        except Exception as e:
            print(f"窦性心律不齐检测失败: {e}")
            return {
                'sinus_arrhythmia_detected': False,
                'severity': '无法评估',
                'hf_ratio': 0.0,
                'lf_hf_ratio': 0.0
            }
    
    def comprehensive_arrhythmia_analysis(self, ppg_signal):
        """
        综合心律不齐分析
        
        Args:
            ppg_signal: PPG信号
            
        Returns:
            analysis_result: 综合分析结果
        """
        try:
            # 信号预处理
            processed_signal, sqi = self.preprocess_signal(ppg_signal)
            
            if processed_signal is None or sqi < 0.3:
                return {
                    'signal_quality': sqi,
                    'analysis_status': '信号质量不足',
                    'arrhythmia_detected': False
                }
            
            # 峰值检测
            peaks, _ = self.detect_peaks(processed_signal)
            
            if len(peaks) < 5:
                return {
                    'signal_quality': sqi,
                    'analysis_status': '峰值检测不足',
                    'arrhythmia_detected': False
                }
            
            # PPI提取
            ppi_seconds, ppi_ms = self.extract_ppi_sequence(peaks)
            
            if len(ppi_ms) < 3:
                return {
                    'signal_quality': sqi,
                    'analysis_status': 'PPI序列过短',
                    'arrhythmia_detected': False
                }
            
            # 特征计算
            time_features = self.calculate_time_domain_features(ppi_ms)
            freq_features = self.calculate_frequency_domain_features(ppi_seconds)
            
            # 各类心律不齐检测
            afib_result = self.detect_atrial_fibrillation(ppi_ms, time_features)
            premature_beats = self.detect_premature_beats(ppi_ms)
            sinus_arrhythmia = self.detect_sinus_arrhythmia(freq_features)
            
            # 综合评估
            arrhythmia_detected = (afib_result['afib_detected'] or 
                                 premature_beats['total_premature'] > 0 or
                                 sinus_arrhythmia['sinus_arrhythmia_detected'])
            
            # 风险评估
            risk_assessment = self._assess_arrhythmia_risk(
                afib_result, premature_beats, sinus_arrhythmia, time_features
            )
            
            analysis_result = {
                'signal_quality': float(sqi),
                'analysis_status': '分析完成',
                'arrhythmia_detected': arrhythmia_detected,
                'peak_count': len(peaks),
                'ppi_count': len(ppi_ms),
                'time_domain_features': time_features,
                'frequency_domain_features': freq_features,
                'atrial_fibrillation': afib_result,
                'premature_beats': premature_beats,
                'sinus_arrhythmia': sinus_arrhythmia,
                'risk_assessment': risk_assessment
            }
            
            return analysis_result
            
        except Exception as e:
            print(f"综合心律不齐分析失败: {e}")
            return {
                'signal_quality': 0.0,
                'analysis_status': f'分析失败: {str(e)}',
                'arrhythmia_detected': False
            }
    
    def _assess_arrhythmia_risk(self, afib_result, premature_beats, sinus_arrhythmia, time_features):
        """评估心律不齐风险"""
        risk_score = 0
        risk_factors = []
        
        # 房颤风险
        if afib_result['afib_detected']:
            if afib_result['risk_level'] == '高度疑似房颤':
                risk_score += 5
                risk_factors.append('高度疑似房颤')
            elif afib_result['risk_level'] == '疑似房颤':
                risk_score += 3
                risk_factors.append('疑似房颤')
        
        # 早搏风险
        premature_rate = premature_beats.get('premature_rate', 0)
        if premature_rate > 10:
            risk_score += 3
            risk_factors.append('频发早搏')
        elif premature_rate > 5:
            risk_score += 2
            risk_factors.append('偶发早搏')
        
        # 心率变异性风险
        cv = time_features.get('cv', 0)
        if cv > 20:
            risk_score += 2
            risk_factors.append('心率变异性显著增加')
        elif cv < 5:
            risk_score += 1
            risk_factors.append('心率变异性降低')
        
        # 风险等级
        if risk_score >= 6:
            risk_level = '高风险'
            recommendations = [
                '建议立即就医进行详细心电图检查',
                '避免剧烈运动和情绪激动',
                '密切监测心律变化'
            ]
        elif risk_score >= 3:
            risk_level = '中等风险'
            recommendations = [
                '建议定期监测心律',
                '如有不适症状及时就医',
                '保持健康生活方式'
            ]
        elif risk_score >= 1:
            risk_level = '低风险'
            recommendations = [
                '继续观察心律变化',
                '保持规律作息',
                '适量运动'
            ]
        else:
            risk_level = '正常'
            recommendations = [
                '心律正常，继续保持',
                '定期健康检查'
            ]
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendations': recommendations
        }