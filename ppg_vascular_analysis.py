#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPG血管功能评价分析系统
基于PPG信号进行血管功能评估，包括PWV、AIx、波形形态学指标等
"""

import json
import numpy as np
import os
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

class PPGVascularAnalyzer:
    def __init__(self, sampling_rate=125):
        """
        初始化PPG血管功能分析器
        
        Args:
            sampling_rate: 采样率，默认125Hz
        """
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
        
    def preprocess_signal(self, ppg_data):
        """
        PPG信号预处理
        
        Args:
            ppg_data: PPG原始信号数据
            
        Returns:
            filtered_signal: 滤波后的信号
        """
        if len(ppg_data) < 100:
            return None
            
        # 转换为numpy数组
        signal_data = np.array(ppg_data, dtype=float)
        
        # 去除异常值
        q75, q25 = np.percentile(signal_data, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        signal_data = np.clip(signal_data, lower_bound, upper_bound)
        
        # 带通滤波 (0.5-8Hz)
        try:
            low_freq = 0.5 / self.nyquist
            high_freq = 8.0 / self.nyquist
            b, a = butter(4, [low_freq, high_freq], btype='band')
            filtered_signal = filtfilt(b, a, signal_data)
            return filtered_signal
        except:
            return signal_data
    
    def detect_peaks_and_valleys(self, signal_data):
        """
        检测PPG信号的峰值和谷值
        
        Args:
            signal_data: 预处理后的PPG信号
            
        Returns:
            peaks: 峰值位置
            valleys: 谷值位置
        """
        # 检测峰值
        peaks, _ = find_peaks(signal_data, 
                             height=np.mean(signal_data),
                             distance=int(0.4 * self.sampling_rate))  # 最小间距0.4秒
        
        # 检测谷值
        valleys, _ = find_peaks(-signal_data,
                               height=-np.mean(signal_data),
                               distance=int(0.4 * self.sampling_rate))
        
        return peaks, valleys
    
    def calculate_heart_rate(self, peaks):
        """
        计算心率
        
        Args:
            peaks: 峰值位置数组
            
        Returns:
            heart_rate: 心率 (bpm)
        """
        if len(peaks) < 2:
            return None
            
        # 计算RR间期
        rr_intervals = np.diff(peaks) / self.sampling_rate
        
        # 过滤异常RR间期
        valid_rr = rr_intervals[(rr_intervals > 0.4) & (rr_intervals < 2.0)]
        
        if len(valid_rr) == 0:
            return None
            
        # 计算心率
        heart_rate = 60.0 / np.mean(valid_rr)
        return heart_rate
    
    def calculate_augmentation_index(self, signal_data, peaks):
        """
        计算增强指数 (Augmentation Index, AIx)
        
        Args:
            signal_data: PPG信号数据
            peaks: 峰值位置
            
        Returns:
            aix: 增强指数 (%)
        """
        if len(peaks) < 2:
            return None
            
        aix_values = []
        
        for i in range(len(peaks) - 1):
            start_idx = peaks[i]
            end_idx = peaks[i + 1]
            
            if end_idx - start_idx < 50:  # 太短的周期跳过
                continue
                
            # 提取单个心动周期
            cycle_signal = signal_data[start_idx:end_idx]
            
            # 寻找收缩期前向波峰值 (P1) 和反射波峰值 (P2)
            cycle_peaks, _ = find_peaks(cycle_signal, height=np.mean(cycle_signal))
            
            if len(cycle_peaks) >= 2:
                # P1是第一个峰值，P2是第二个峰值
                p1_idx = cycle_peaks[0]
                p2_idx = cycle_peaks[1]
                
                p1_value = cycle_signal[p1_idx]
                p2_value = cycle_signal[p2_idx]
                
                # 计算脉压 (PP)
                cycle_min = np.min(cycle_signal)
                pulse_pressure = p1_value - cycle_min
                
                if pulse_pressure > 0:
                    # AIx = (P2-P1)/PP × 100%
                    aix = ((p2_value - p1_value) / pulse_pressure) * 100
                    aix_values.append(aix)
        
        if len(aix_values) > 0:
            return np.mean(aix_values)
        else:
            return None
    
    def calculate_morphological_features(self, signal_data, peaks, valleys):
        """
        计算波形形态学指标
        
        Args:
            signal_data: PPG信号数据
            peaks: 峰值位置
            valleys: 谷值位置
            
        Returns:
            morphological_features: 形态学特征字典
        """
        features = {
            'dicrotic_notch_index': None,
            'rise_time_ratio': None,
            'systolic_diastolic_ratio': None,
            'pulse_width': None,
            'amplitude_variation': None
        }
        
        if len(peaks) < 2 or len(valleys) < 2:
            return features
        
        # 计算上升时间比
        rise_times = []
        pulse_widths = []
        systolic_areas = []
        diastolic_areas = []
        
        for i in range(min(len(peaks), len(valleys)) - 1):
            peak_idx = peaks[i]
            valley_start = valleys[i] if i < len(valleys) else valleys[-1]
            valley_end = valleys[i + 1] if i + 1 < len(valleys) else len(signal_data) - 1
            
            # 上升时间
            rise_time = (peak_idx - valley_start) / self.sampling_rate
            rise_times.append(rise_time)
            
            # 脉搏宽度
            pulse_width = (valley_end - valley_start) / self.sampling_rate
            pulse_widths.append(pulse_width)
            
            # 收缩期和舒张期面积
            if valley_end > peak_idx:
                systolic_signal = signal_data[valley_start:peak_idx]
                diastolic_signal = signal_data[peak_idx:valley_end]
                
                systolic_area = np.trapz(systolic_signal)
                diastolic_area = np.trapz(diastolic_signal)
                
                systolic_areas.append(systolic_area)
                diastolic_areas.append(diastolic_area)
        
        # 计算特征值
        if len(rise_times) > 0 and len(pulse_widths) > 0:
            features['rise_time_ratio'] = np.mean(rise_times) / np.mean(pulse_widths)
            features['pulse_width'] = np.mean(pulse_widths)
        
        if len(systolic_areas) > 0 and len(diastolic_areas) > 0:
            features['systolic_diastolic_ratio'] = np.mean(systolic_areas) / np.mean(diastolic_areas)
        
        # 振幅变异性
        if len(peaks) > 1:
            peak_amplitudes = signal_data[peaks]
            features['amplitude_variation'] = np.std(peak_amplitudes) / np.mean(peak_amplitudes)
        
        return features
    
    def calculate_sdppg_features(self, signal_data):
        """
        计算二阶导数分析 (SDPPG) 特征
        
        Args:
            signal_data: PPG信号数据
            
        Returns:
            sdppg_features: SDPPG特征字典
        """
        # 计算一阶导数
        first_derivative = np.gradient(signal_data)
        
        # 计算二阶导数
        second_derivative = np.gradient(first_derivative)
        
        # 寻找SDPPG特征波
        # a波：正向峰值
        a_peaks, _ = find_peaks(second_derivative, height=0)
        
        # b波、c波、d波：负向峰值
        negative_peaks, _ = find_peaks(-second_derivative, height=0)
        
        features = {
            'a_wave_amplitude': None,
            'b_wave_amplitude': None,
            'c_wave_amplitude': None,
            'd_wave_amplitude': None,
            'aging_index': None  # (b-c-d)/a
        }
        
        if len(a_peaks) > 0:
            features['a_wave_amplitude'] = np.mean(second_derivative[a_peaks])
        
        if len(negative_peaks) >= 3:
            # 按时间顺序排序负峰
            sorted_negative = np.sort(negative_peaks)
            
            features['b_wave_amplitude'] = -second_derivative[sorted_negative[0]]
            features['c_wave_amplitude'] = -second_derivative[sorted_negative[1]]
            features['d_wave_amplitude'] = -second_derivative[sorted_negative[2]]
            
            # 计算血管老化指数
            if features['a_wave_amplitude'] and features['a_wave_amplitude'] > 0:
                aging_index = (features['b_wave_amplitude'] - 
                             features['c_wave_amplitude'] - 
                             features['d_wave_amplitude']) / features['a_wave_amplitude']
                features['aging_index'] = aging_index
        
        return features
    
    def calculate_frequency_domain_features(self, signal_data):
        """
        计算频域分析特征
        
        Args:
            signal_data: PPG信号数据
            
        Returns:
            frequency_features: 频域特征字典
        """
        # FFT分析
        fft_values = fft(signal_data)
        frequencies = fftfreq(len(signal_data), 1/self.sampling_rate)
        
        # 只取正频率部分
        positive_freq_idx = frequencies > 0
        frequencies = frequencies[positive_freq_idx]
        power_spectrum = np.abs(fft_values[positive_freq_idx])**2
        
        # 定义频段
        vlf_band = (frequencies >= 0.003) & (frequencies < 0.04)  # 极低频
        lf_band = (frequencies >= 0.04) & (frequencies < 0.15)   # 低频
        hf_band = (frequencies >= 0.15) & (frequencies < 0.4)    # 高频
        
        # 计算各频段功率
        vlf_power = np.sum(power_spectrum[vlf_band])
        lf_power = np.sum(power_spectrum[lf_band])
        hf_power = np.sum(power_spectrum[hf_band])
        total_power = vlf_power + lf_power + hf_power
        
        features = {
            'vlf_power': vlf_power,
            'lf_power': lf_power,
            'hf_power': hf_power,
            'total_power': total_power,
            'lf_hf_ratio': lf_power / hf_power if hf_power > 0 else None,
            'normalized_lf': lf_power / total_power if total_power > 0 else None,
            'normalized_hf': hf_power / total_power if total_power > 0 else None
        }
        
        return features
    
    def estimate_pwv(self, signal_data, heart_rate):
        """
        估算脉搏波传导速度 (PWV)
        基于单点PPG信号的PWV估算方法
        
        Args:
            signal_data: PPG信号数据
            heart_rate: 心率
            
        Returns:
            estimated_pwv: 估算的PWV值 (m/s)
        """
        if heart_rate is None:
            return None
        
        # 基于心率和波形特征的PWV估算
        # 这是一个简化的估算方法，实际PWV需要多点测量
        
        # 计算波形的陡峭度（上升斜率）
        peaks, valleys = self.detect_peaks_and_valleys(signal_data)
        
        if len(peaks) < 2 or len(valleys) < 2:
            return None
        
        slopes = []
        for i in range(min(len(peaks), len(valleys))):
            peak_idx = peaks[i]
            valley_idx = valleys[i] if i < len(valleys) else valleys[-1]
            
            if peak_idx > valley_idx:
                rise_signal = signal_data[valley_idx:peak_idx]
                if len(rise_signal) > 1:
                    slope = np.max(np.gradient(rise_signal))
                    slopes.append(slope)
        
        if len(slopes) == 0:
            return None
        
        mean_slope = np.mean(slopes)
        
        # 基于经验公式估算PWV
        # PWV与心率和波形陡峭度相关
        estimated_pwv = 5.0 + (heart_rate - 70) * 0.05 + mean_slope * 0.1
        
        # 限制在合理范围内
        estimated_pwv = np.clip(estimated_pwv, 4.0, 20.0)
        
        return estimated_pwv
    
    def assess_vascular_age(self, pwv, aix, heart_rate, chronological_age=None):
        """
        评估血管年龄
        
        Args:
            pwv: 脉搏波传导速度
            aix: 增强指数
            heart_rate: 心率
            chronological_age: 实际年龄（如果已知）
            
        Returns:
            vascular_age: 血管年龄
        """
        if pwv is None and aix is None:
            return None
        
        # 基于PWV的血管年龄估算
        age_from_pwv = None
        if pwv is not None:
            # 基于文献的PWV-年龄关系
            age_from_pwv = (pwv - 5.0) * 10 + 30
            age_from_pwv = np.clip(age_from_pwv, 20, 90)
        
        # 基于AIx的血管年龄估算
        age_from_aix = None
        if aix is not None:
            # AIx随年龄增加
            age_from_aix = aix * 2 + 30
            age_from_aix = np.clip(age_from_aix, 20, 90)
        
        # 综合评估
        if age_from_pwv is not None and age_from_aix is not None:
            vascular_age = (age_from_pwv + age_from_aix) / 2
        elif age_from_pwv is not None:
            vascular_age = age_from_pwv
        elif age_from_aix is not None:
            vascular_age = age_from_aix
        else:
            return None
        
        return vascular_age
    
    def analyze_ppg_segment(self, ppg_data):
        """
        分析单个PPG数据段
        
        Args:
            ppg_data: PPG原始数据
            
        Returns:
            analysis_result: 分析结果字典
        """
        # 预处理信号
        filtered_signal = self.preprocess_signal(ppg_data)
        if filtered_signal is None:
            return None
        
        # 检测峰值和谷值
        peaks, valleys = self.detect_peaks_and_valleys(filtered_signal)
        
        # 计算心率
        heart_rate = self.calculate_heart_rate(peaks)
        
        # 计算增强指数
        aix = self.calculate_augmentation_index(filtered_signal, peaks)
        
        # 计算形态学特征
        morphological_features = self.calculate_morphological_features(filtered_signal, peaks, valleys)
        
        # 计算SDPPG特征
        sdppg_features = self.calculate_sdppg_features(filtered_signal)
        
        # 计算频域特征
        frequency_features = self.calculate_frequency_domain_features(filtered_signal)
        
        # 估算PWV
        estimated_pwv = self.estimate_pwv(filtered_signal, heart_rate)
        
        # 评估血管年龄
        vascular_age = self.assess_vascular_age(estimated_pwv, aix, heart_rate)
        
        # 组装结果
        result = {
            'signal_quality': 'good' if heart_rate is not None else 'poor',
            'heart_rate': heart_rate,
            'estimated_pwv': estimated_pwv,
            'augmentation_index': aix,
            'vascular_age': vascular_age,
            'morphological_features': morphological_features,
            'sdppg_features': sdppg_features,
            'frequency_domain_features': frequency_features,
            'data_length': len(ppg_data),
            'peaks_count': len(peaks),
            'valleys_count': len(valleys)
        }
        
        return result

def main():
    """主函数，用于测试"""
    analyzer = PPGVascularAnalyzer()
    
    # 测试数据
    test_data = np.random.randn(1000) + np.sin(np.linspace(0, 10*np.pi, 1000))
    
    result = analyzer.analyze_ppg_segment(test_data)
    print("测试结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()