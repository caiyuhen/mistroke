"""
PPG炎症检测模块
基于PPG信号的系统性炎症检测和评估
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class PPGInflammationDetector:
    """PPG炎症检测器"""
    
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate
        self.scaler = StandardScaler()
        self._window_cache = {}
        
    def calculate_inflammation_perfusion_index(self, ppg_signal):
        """计算炎症相关的灌注指标"""
        try:
            # 基础灌注指数计算
            ac_component = np.std(ppg_signal)
            dc_component = np.mean(ppg_signal)
            pi = (ac_component / dc_component) * 100 if dc_component != 0 else 0
            
            # 计算灌注变异系数
            # 使用滑动窗口计算局部灌注指数
            window_size = min(len(ppg_signal) // 10, 500)
            pi_timeseries = []
            
            for i in range(0, len(ppg_signal) - window_size, window_size // 2):
                segment = ppg_signal[i:i + window_size]
                ac_seg = np.std(segment)
                dc_seg = np.mean(segment)
                pi_seg = (ac_seg / dc_seg) * 100 if dc_seg != 0 else 0
                pi_timeseries.append(pi_seg)
            
            pi_timeseries = np.array(pi_timeseries)
            pi_variability = np.std(pi_timeseries) / np.mean(pi_timeseries) if np.mean(pi_timeseries) != 0 else 0
            
            # 微循环阻力指数
            resistance_index = self._calculate_vascular_resistance(ppg_signal)
            
            return {
                'perfusion_index': pi,
                'perfusion_variability': pi_variability * 100,  # 转换为百分比
                'microvascular_resistance': resistance_index,
                'perfusion_quality': self._assess_perfusion_quality(pi, pi_variability)
            }
        except Exception as e:
            return {
                'perfusion_index': 0,
                'perfusion_variability': 0,
                'microvascular_resistance': 0,
                'perfusion_quality': 'poor'
            }
    
    def _calculate_vascular_resistance(self, ppg_signal):
        """计算血管阻力指数"""
        try:
            # 使用PPG信号的上升时间和峰值比来估算血管阻力
            peaks, _ = signal.find_peaks(ppg_signal, height=np.mean(ppg_signal))
            if len(peaks) < 2:
                return 0
            
            # 计算平均上升时间
            rise_times = []
            for peak in peaks[:min(10, len(peaks))]:
                # 找到峰值前的谷值
                start_idx = max(0, peak - 50)
                valley_idx = start_idx + np.argmin(ppg_signal[start_idx:peak])
                rise_time = peak - valley_idx
                rise_times.append(rise_time)
            
            avg_rise_time = np.mean(rise_times) if rise_times else 0
            
            # 计算峰值变异性
            peak_values = ppg_signal[peaks]
            peak_variability = np.std(peak_values) / np.mean(peak_values) if np.mean(peak_values) != 0 else 0
            
            # 阻力指数 = 上升时间 * 峰值变异性
            resistance_index = avg_rise_time * peak_variability
            
            return resistance_index
        except:
            return 0
    
    def _assess_perfusion_quality(self, pi, pi_variability):
        """评估灌注质量"""
        if pi > 0.3 and pi_variability < 0.15:
            return 'excellent'
        elif pi > 0.2 and pi_variability < 0.25:
            return 'good'
        elif pi > 0.1 and pi_variability < 0.35:
            return 'fair'
        else:
            return 'poor'
    
    def extract_inflammation_morphology(self, ppg_waveform):
        """提取炎症相关的波形特征"""
        try:
            features = {}
            
            # 检测峰值和谷值
            peaks, _ = signal.find_peaks(ppg_waveform, height=np.mean(ppg_waveform))
            valleys, _ = signal.find_peaks(-ppg_waveform, height=-np.mean(ppg_waveform))
            
            if len(peaks) < 2 or len(valleys) < 2:
                return self._get_default_morphology_features()
            
            # 上升时间比
            rise_times = []
            cycle_times = []
            
            for i in range(min(len(peaks) - 1, 10)):
                peak_idx = peaks[i]
                next_peak_idx = peaks[i + 1]
                
                # 找到当前峰值前的谷值
                prev_valleys = valleys[valleys < peak_idx]
                if len(prev_valleys) > 0:
                    valley_idx = prev_valleys[-1]
                    rise_time = peak_idx - valley_idx
                    cycle_time = next_peak_idx - peak_idx
                    
                    rise_times.append(rise_time)
                    cycle_times.append(cycle_time)
            
            if rise_times and cycle_times:
                features['rise_time_ratio'] = np.mean(rise_times) / np.mean(cycle_times)
            else:
                features['rise_time_ratio'] = 0.3  # 默认值
            
            # 重搏切迹指数
            features['dicrotic_prominence'] = self._detect_dicrotic_notch(ppg_waveform, peaks)
            
            # 波形对称性
            features['waveform_skewness'] = abs(stats.skew(ppg_waveform))
            
            # 频谱熵
            features['spectral_entropy'] = self._calculate_spectral_entropy(ppg_waveform)
            
            # 波形复杂度
            features['waveform_complexity'] = self._calculate_waveform_complexity(ppg_waveform)
            
            return features
        except Exception as e:
            return self._get_default_morphology_features()
    
    def _get_default_morphology_features(self):
        """返回默认的形态学特征"""
        return {
            'rise_time_ratio': 0.3,
            'dicrotic_prominence': 0.1,
            'waveform_skewness': 0.5,
            'spectral_entropy': 0.5,
            'waveform_complexity': 0.5
        }
    
    def _detect_dicrotic_notch(self, ppg_waveform, peaks):
        """检测重搏切迹"""
        try:
            if len(peaks) < 2:
                return 0.1
            
            dicrotic_ratios = []
            for peak in peaks[:min(5, len(peaks))]:
                # 在峰值后寻找重搏切迹
                search_start = peak
                search_end = min(peak + 50, len(ppg_waveform))
                
                if search_end > search_start:
                    segment = ppg_waveform[search_start:search_end]
                    min_idx = np.argmin(segment)
                    
                    if min_idx > 0 and min_idx < len(segment) - 1:
                        notch_depth = ppg_waveform[peak] - segment[min_idx]
                        peak_amplitude = ppg_waveform[peak] - np.min(ppg_waveform)
                        
                        if peak_amplitude > 0:
                            dicrotic_ratio = notch_depth / peak_amplitude
                            dicrotic_ratios.append(dicrotic_ratio)
            
            return np.mean(dicrotic_ratios) if dicrotic_ratios else 0.1
        except:
            return 0.1
    
    def _calculate_spectral_entropy(self, ppg_waveform):
        """计算频谱熵"""
        try:
            # 计算功率谱密度
            nperseg = min(256, max(8, len(ppg_waveform)//4))
            w = self._window_cache.get(nperseg)
            if w is None:
                w = signal.get_window('hann', nperseg)
                self._window_cache[nperseg] = w
            freqs, psd = signal.welch(ppg_waveform, fs=self.sampling_rate, nperseg=nperseg, window=w)
            
            # 归一化功率谱
            psd_norm = psd / np.sum(psd)
            
            # 计算熵
            entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
            
            # 归一化到0-1范围
            max_entropy = np.log2(len(psd_norm))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            return normalized_entropy
        except:
            return 0.5
    
    def _calculate_waveform_complexity(self, ppg_waveform):
        try:
            fast_fd = os.environ.get('FAST_FD', '0') == '1'
            if fast_fd:
                x = np.asarray(ppg_waveform, dtype=np.float64)
                diff = np.diff(x)
                N = x.size
                if N < 2:
                    return 0.5
                # Petrosian FD
                sign_changes = np.sum(diff[1:] * diff[:-1] < 0)
                v = sign_changes
                return min(max((np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * v)))), 0), 2) / 2
            def higuchi_fd(signal_data, k_max=10):
                N = len(signal_data)
                L = []

                for k in range(1, k_max + 1):
                    Lk = []
                    for m in range(k):
                        s = signal_data[m::k]
                        n = s.size
                        if n < 2:
                            continue
                        diffs = np.abs(np.diff(s))
                        Lmk = diffs.sum() * (N - 1) / (k * k * (n - 1))
                        Lk.append(Lmk)
                    L.append(np.mean(Lk) if Lk else 0.0)

                x = np.log(np.arange(1, k_max + 1))
                y = np.log(np.array(L) + 1e-12)

                if x.size > 1 and y.size > 1:
                    slope, _ = np.polyfit(x, y, 1)
                    return -slope
                else:
                    return 1.5
            k_cfg = os.environ.get('INFLAM_K_MAX')
            k_val = int(k_cfg) if k_cfg and k_cfg.isdigit() else 10
            ds_cfg = os.environ.get('INFLAM_DOWNSAMPLE')
            ds = int(ds_cfg) if ds_cfg and ds_cfg.isdigit() and int(ds_cfg) > 1 else 1
            data = ppg_waveform[::ds] if ds > 1 else ppg_waveform
            fd = higuchi_fd(data, k_max=k_val)
            # 归一化到0-1范围
            complexity = (fd - 1.0) / 1.0 if fd > 1.0 else 0
            return min(max(complexity, 0), 1)
        except:
            return 0.5
    
    def inflammation_hrv_analysis(self, ppg_signal):
        """基于PPG的炎症相关HRV分析"""
        try:
            # 提取峰值间期（类似RR间期）
            peak_intervals = self._extract_peak_intervals(ppg_signal)
            
            if len(peak_intervals) < 5:
                return self._get_default_hrv_features()
            
            # 时域指标
            sdnn = np.std(peak_intervals)  # 标准差
            rmssd = self._calculate_rmssd(peak_intervals)  # 相邻间期差值的均方根
            
            # 频域指标
            lf_power, hf_power = self._calculate_frequency_power(peak_intervals)
            lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 5.0
            
            # 非线性指标
            pnn50 = self._calculate_pnn50(peak_intervals)
            
            return {
                'sdnn': sdnn,
                'rmssd': rmssd,
                'lf_hf_ratio': lf_hf_ratio,
                'total_power': lf_power + hf_power,
                'pnn50': pnn50,
                'hrv_quality': self._assess_hrv_quality(sdnn, rmssd, lf_hf_ratio)
            }
        except Exception as e:
            return self._get_default_hrv_features()
    
    def _extract_peak_intervals(self, ppg_signal):
        """提取峰值间期"""
        try:
            # 检测峰值
            peaks, _ = signal.find_peaks(ppg_signal, 
                                       height=np.mean(ppg_signal),
                                       distance=int(0.4 * self.sampling_rate))  # 最小间隔0.4秒
            
            if len(peaks) < 2:
                return []
            
            # 计算间期（转换为毫秒）
            intervals = np.diff(peaks) * (1000 / self.sampling_rate)
            
            # 过滤异常值
            median_interval = np.median(intervals)
            valid_intervals = intervals[(intervals > 0.3 * median_interval) & 
                                     (intervals < 3.0 * median_interval)]
            
            return valid_intervals
        except:
            return []
    
    def _calculate_rmssd(self, intervals):
        """计算RMSSD"""
        if len(intervals) < 2:
            return 0
        
        diff_intervals = np.diff(intervals)
        rmssd = np.sqrt(np.mean(diff_intervals ** 2))
        return rmssd
    
    def _calculate_pnn50(self, intervals):
        """计算pNN50"""
        if len(intervals) < 2:
            return 0
        
        diff_intervals = np.abs(np.diff(intervals))
        pnn50 = np.sum(diff_intervals > 50) / len(diff_intervals) * 100
        return pnn50
    
    def _calculate_frequency_power(self, intervals):
        """计算频域功率"""
        try:
            if len(intervals) < 10:
                return 0, 0
            
            # 重采样到均匀时间间隔
            time_points = np.cumsum(intervals) / 1000.0  # 转换为秒
            fs_resample = 4.0  # 4Hz重采样
            
            # 创建均匀时间序列
            time_uniform = np.arange(0, time_points[-1], 1/fs_resample)
            intervals_uniform = np.interp(time_uniform, time_points[:-1], intervals[:-1])
            
            # 去趋势
            intervals_detrend = signal.detrend(intervals_uniform)
            
            # 计算功率谱密度
            freqs, psd = signal.welch(intervals_detrend, fs=fs_resample, nperseg=min(256, len(intervals_detrend)//4))
            
            # 定义频带
            lf_band = (freqs >= 0.04) & (freqs <= 0.15)
            hf_band = (freqs >= 0.15) & (freqs <= 0.4)
            
            lf_power = np.trapz(psd[lf_band], freqs[lf_band])
            hf_power = np.trapz(psd[hf_band], freqs[hf_band])
            
            return lf_power, hf_power
        except:
            return 0, 0
    
    def _get_default_hrv_features(self):
        """返回默认的HRV特征"""
        return {
            'sdnn': 0,
            'rmssd': 0,
            'lf_hf_ratio': 2.0,
            'total_power': 0,
            'pnn50': 0,
            'hrv_quality': 'poor'
        }
    
    def _assess_hrv_quality(self, sdnn, rmssd, lf_hf_ratio):
        """评估HRV质量"""
        if sdnn > 50 and rmssd > 30 and 0.5 <= lf_hf_ratio <= 3.0:
            return 'excellent'
        elif sdnn > 30 and rmssd > 20 and 0.3 <= lf_hf_ratio <= 5.0:
            return 'good'
        elif sdnn > 15 and rmssd > 10:
            return 'fair'
        else:
            return 'poor'
    
    def ppg_inflammation_score(self, ppg_signal, patient_info=None):
        """计算PPG炎症风险评分"""
        try:
            # 特征提取
            perfusion_features = self.calculate_inflammation_perfusion_index(ppg_signal)
            morphology_features = self.extract_inflammation_morphology(ppg_signal)
            hrv_features = self.inflammation_hrv_analysis(ppg_signal)
            
            # 构建特征向量
            feature_vector = np.array([
                perfusion_features['perfusion_index'],
                perfusion_features['perfusion_variability'] / 100,  # 归一化
                morphology_features['rise_time_ratio'],
                morphology_features['dicrotic_prominence'],
                morphology_features['spectral_entropy'],
                hrv_features['sdnn'] / 100,  # 归一化
                hrv_features['rmssd'] / 100,  # 归一化
                min(hrv_features['lf_hf_ratio'] / 10, 1.0)  # 归一化并限制上限
            ])
            
            # 权重分配（基于临床相关性）
            weights = np.array([0.2, 0.15, 0.15, 0.1, 0.1, 0.15, 0.1, 0.05])
            
            # 炎症评分计算
            # 对于炎症指标，值越低风险越高
            risk_vector = np.array([
                1 - min(feature_vector[0] / 0.5, 1.0),  # 灌注指数越低风险越高
                min(feature_vector[1] * 4, 1.0),        # 灌注变异性越高风险越高
                abs(feature_vector[2] - 0.3) * 3,       # 偏离正常上升时间比
                1 - feature_vector[3],                   # 重搏切迹越弱风险越高
                feature_vector[4],                       # 频谱熵越高风险越高
                1 - min(feature_vector[5] * 2, 1.0),    # SDNN越低风险越高
                1 - min(feature_vector[6] * 3, 1.0),    # RMSSD越低风险越高
                min((feature_vector[7] - 0.2) * 2, 1.0) # LF/HF比值偏离正常范围
            ])
            
            # 确保所有值在0-1范围内
            risk_vector = np.clip(risk_vector, 0, 1)
            
            # 计算加权评分
            inflammation_score = np.dot(risk_vector, weights) * 10  # 转换为0-10分
            
            # 评估炎症等级
            inflammation_grade = self._classify_inflammation_grade(inflammation_score)
            
            return {
                'inflammation_score': inflammation_score,
                'inflammation_grade': inflammation_grade,
                'risk_level': self._assess_risk_level(inflammation_score),
                'feature_contributions': {
                    'perfusion_risk': risk_vector[0] * weights[0] * 10,
                    'variability_risk': risk_vector[1] * weights[1] * 10,
                    'morphology_risk': (risk_vector[2] + risk_vector[3] + risk_vector[4]) * sum(weights[2:5]) * 10,
                    'hrv_risk': (risk_vector[5] + risk_vector[6] + risk_vector[7]) * sum(weights[5:8]) * 10
                }
            }
        except Exception as e:
            return {
                'inflammation_score': 5.0,
                'inflammation_grade': 'moderate',
                'risk_level': 'moderate',
                'feature_contributions': {
                    'perfusion_risk': 1.25,
                    'variability_risk': 1.25,
                    'morphology_risk': 1.25,
                    'hrv_risk': 1.25
                }
            }
    
    def _classify_inflammation_grade(self, score):
        """分类炎症等级"""
        if score <= 2:
            return 'normal'
        elif score <= 4:
            return 'mild'
        elif score <= 6:
            return 'moderate'
        else:
            return 'severe'
    
    def _assess_risk_level(self, score):
        """评估风险等级"""
        if score <= 2:
            return 'low'
        elif score <= 4:
            return 'moderate'
        elif score <= 6:
            return 'high'
        else:
            return 'critical'
    
    def inflammation_early_warning(self, ppg_timeseries_list, baseline_days=7):
        """基于连续PPG监测的炎症早期预警"""
        try:
            if len(ppg_timeseries_list) < 2:
                return {'alert_level': 'INSUFFICIENT_DATA'}
            
            # 计算基线特征（使用前几天的数据）
            baseline_scores = []
            current_scores = []
            
            # 基线期数据
            baseline_data = ppg_timeseries_list[:min(baseline_days, len(ppg_timeseries_list)//2)]
            for ppg_data in baseline_data:
                score_result = self.ppg_inflammation_score(ppg_data)
                baseline_scores.append(score_result['inflammation_score'])
            
            # 当前期数据
            current_data = ppg_timeseries_list[-min(3, len(ppg_timeseries_list)//4):]
            for ppg_data in current_data:
                score_result = self.ppg_inflammation_score(ppg_data)
                current_scores.append(score_result['inflammation_score'])
            
            if not baseline_scores or not current_scores:
                return {'alert_level': 'INSUFFICIENT_DATA'}
            
            # 计算趋势
            baseline_mean = np.mean(baseline_scores)
            current_mean = np.mean(current_scores)
            
            # 计算变化率
            change_rate = (current_mean - baseline_mean) / baseline_mean if baseline_mean > 0 else 0
            
            # 计算趋势斜率
            all_scores = baseline_scores + current_scores
            time_points = np.arange(len(all_scores))
            if len(all_scores) > 2:
                slope, _ = np.polyfit(time_points, all_scores, 1)
            else:
                slope = 0
            
            # 风险评估
            inflammation_risk = self._calculate_inflammation_risk(current_mean, change_rate, slope)
            
            # 预警等级判断
            if inflammation_risk > 0.7:
                return {
                    'alert_level': 'HIGH',
                    'inflammation_risk': inflammation_risk,
                    'current_score': current_mean,
                    'baseline_score': baseline_mean,
                    'change_rate': change_rate * 100,
                    'trend_slope': slope,
                    'predicted_onset': '24-48小时内',
                    'recommended_action': '建议血液检查确认炎症标志物'
                }
            elif inflammation_risk > 0.4:
                return {
                    'alert_level': 'MODERATE',
                    'inflammation_risk': inflammation_risk,
                    'current_score': current_mean,
                    'baseline_score': baseline_mean,
                    'change_rate': change_rate * 100,
                    'trend_slope': slope,
                    'predicted_onset': '3-5天内',
                    'recommended_action': '密切观察，注意症状变化'
                }
            else:
                return {
                    'alert_level': 'LOW',
                    'inflammation_risk': inflammation_risk,
                    'current_score': current_mean,
                    'baseline_score': baseline_mean,
                    'change_rate': change_rate * 100,
                    'trend_slope': slope
                }
        except Exception as e:
            return {'alert_level': 'ERROR', 'error': str(e)}
    
    def _calculate_inflammation_risk(self, current_score, change_rate, slope):
        """计算炎症风险"""
        # 基于当前评分的风险
        score_risk = min(current_score / 10, 1.0)
        
        # 基于变化率的风险
        change_risk = min(abs(change_rate), 1.0) if change_rate > 0 else 0
        
        # 基于趋势的风险
        trend_risk = min(abs(slope) / 2, 1.0) if slope > 0 else 0
        
        # 综合风险评估
        total_risk = (score_risk * 0.5 + change_risk * 0.3 + trend_risk * 0.2)
        
        return min(total_risk, 1.0)
    
    def comprehensive_inflammation_analysis(self, ppg_signal):
        """综合炎症分析"""
        try:
            # 基础特征提取
            perfusion_features = self.calculate_inflammation_perfusion_index(ppg_signal)
            morphology_features = self.extract_inflammation_morphology(ppg_signal)
            hrv_features = self.inflammation_hrv_analysis(ppg_signal)
            
            # 炎症评分
            inflammation_result = self.ppg_inflammation_score(ppg_signal)
            
            # 预测炎症标志物水平（基于经验模型）
            predicted_markers = self._predict_inflammation_markers(
                perfusion_features, morphology_features, hrv_features
            )
            
            # 临床建议
            clinical_recommendations = self._generate_clinical_recommendations(
                inflammation_result['inflammation_grade'],
                inflammation_result['risk_level']
            )
            
            return {
                'perfusion_analysis': perfusion_features,
                'morphology_analysis': morphology_features,
                'hrv_analysis': hrv_features,
                'inflammation_assessment': inflammation_result,
                'predicted_markers': predicted_markers,
                'clinical_recommendations': clinical_recommendations,
                'analysis_quality': self._assess_analysis_quality(
                    perfusion_features, morphology_features, hrv_features
                )
            }
        except Exception as e:
            return {
                'error': str(e),
                'analysis_quality': 'poor'
            }
    
    def _predict_inflammation_markers(self, perfusion_features, morphology_features, hrv_features):
        """预测炎症标志物水平"""
        try:
            # 基于文献相关性的经验模型
            
            # CRP预测（与灌注指数负相关）
            crp_level = max(0, 10 - perfusion_features['perfusion_index'] * 20)
            
            # IL-6预测（与HRV参数负相关）
            il6_level = max(0, 50 - hrv_features['rmssd'] * 2)
            
            # TNF-α预测（与波形复杂度正相关）
            tnf_level = morphology_features['spectral_entropy'] * 100
            
            # ESR预测（与微循环阻力正相关）
            esr_level = min(100, perfusion_features['microvascular_resistance'] * 10)
            
            return {
                'predicted_crp': round(crp_level, 2),
                'predicted_il6': round(il6_level, 2),
                'predicted_tnf_alpha': round(tnf_level, 2),
                'predicted_esr': round(esr_level, 2),
                'prediction_confidence': self._calculate_prediction_confidence(
                    perfusion_features, morphology_features, hrv_features
                )
            }
        except:
            return {
                'predicted_crp': 0,
                'predicted_il6': 0,
                'predicted_tnf_alpha': 0,
                'predicted_esr': 0,
                'prediction_confidence': 'low'
            }
    
    def _calculate_prediction_confidence(self, perfusion_features, morphology_features, hrv_features):
        """计算预测置信度"""
        quality_scores = [
            1 if perfusion_features['perfusion_quality'] in ['excellent', 'good'] else 0,
            1 if hrv_features['hrv_quality'] in ['excellent', 'good'] else 0,
            1 if morphology_features['spectral_entropy'] > 0.3 else 0
        ]
        
        confidence_score = sum(quality_scores) / len(quality_scores)
        
        if confidence_score >= 0.8:
            return 'high'
        elif confidence_score >= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _generate_clinical_recommendations(self, inflammation_grade, risk_level):
        """生成临床建议"""
        recommendations = {
            'normal': {
                'monitoring': '常规监测即可',
                'lifestyle': '保持健康生活方式',
                'follow_up': '3-6个月复查'
            },
            'mild': {
                'monitoring': '增加监测频率',
                'lifestyle': '注意休息，避免过度劳累',
                'follow_up': '1-2个月复查',
                'additional': '可考虑抗氧化剂补充'
            },
            'moderate': {
                'monitoring': '密切监测',
                'lifestyle': '充分休息，避免感染',
                'follow_up': '2-4周复查',
                'additional': '建议血液检查确认炎症标志物'
            },
            'severe': {
                'monitoring': '持续监测',
                'lifestyle': '卧床休息，避免剧烈活动',
                'follow_up': '1周内复查',
                'additional': '立即进行全面炎症评估，考虑抗炎治疗'
            }
        }
        
        return recommendations.get(inflammation_grade, recommendations['moderate'])
    
    def _assess_analysis_quality(self, perfusion_features, morphology_features, hrv_features):
        """评估分析质量"""
        quality_indicators = [
            perfusion_features['perfusion_quality'],
            hrv_features['hrv_quality'],
            'good' if morphology_features['spectral_entropy'] > 0.3 else 'poor'
        ]
        
        excellent_count = sum(1 for q in quality_indicators if q == 'excellent')
        good_count = sum(1 for q in quality_indicators if q == 'good')
        
        if excellent_count >= 2:
            return 'excellent'
        elif excellent_count + good_count >= 2:
            return 'good'
        elif good_count >= 1:
            return 'fair'
        else:
            return 'poor'