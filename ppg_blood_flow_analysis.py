#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPG血流分析模块
基于PPG信号间接估算血流相关参数，包括灌注指数、血流动力学建模等
"""

import numpy as np
import scipy.stats
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

class PPGBloodFlowEstimator:
    def __init__(self, sampling_rate=125):
        """
        初始化PPG血流估算器
        
        Args:
            sampling_rate: 采样率，默认125Hz
        """
        self.sampling_rate = sampling_rate
        self.baseline_pi = None
        self.calibration_factor = 1.0
        
    def calculate_perfusion_index(self, ppg_signal):
        """
        计算灌注指数，反映局部血流灌注
        
        Args:
            ppg_signal: PPG信号数据
            
        Returns:
            PI: 灌注指数 (%)
            flow_status: 血流状态
        """
        if len(ppg_signal) < 10:
            return None, "数据不足"
        
        # 计算脉动成分(AC)和直流成分(DC)
        ac_component = np.std(ppg_signal)  # 脉动成分
        dc_component = np.mean(ppg_signal)  # 直流成分
        
        if dc_component == 0:
            return None, "信号异常"
        
        # 计算灌注指数
        PI = (ac_component / abs(dc_component)) * 100
        
        # 血流状态判断
        if PI > 5.0:
            flow_status = "高灌注"
        elif PI > 0.3:
            flow_status = "正常灌注"
        else:
            flow_status = "低灌注"
            
        return PI, flow_status
    
    def estimate_compliance(self, age=None, bmi=None, systolic_bp=None):
        """
        估算血管顺应性
        
        Args:
            age: 年龄
            bmi: 体重指数
            systolic_bp: 收缩压
            
        Returns:
            compliance: 血管顺应性估算值
        """
        # 基础顺应性值
        base_compliance = 1.5
        
        # 年龄修正
        if age is not None:
            age_factor = max(0.5, 1.0 - (age - 30) * 0.01)
            base_compliance *= age_factor
        
        # BMI修正
        if bmi is not None:
            if bmi > 25:
                bmi_factor = max(0.7, 1.0 - (bmi - 25) * 0.02)
                base_compliance *= bmi_factor
        
        # 血压修正
        if systolic_bp is not None:
            if systolic_bp > 120:
                bp_factor = max(0.6, 1.0 - (systolic_bp - 120) * 0.005)
                base_compliance *= bp_factor
        
        return base_compliance
    
    def estimate_resistance(self, ppg_waveform):
        """
        基于PPG波形估算血管阻力
        
        Args:
            ppg_waveform: PPG波形数据
            
        Returns:
            resistance: 血管阻力估算值
        """
        if len(ppg_waveform) < 50:
            return 1.0
        
        # 计算波形的陡峭度
        gradient = np.gradient(ppg_waveform)
        max_gradient = np.max(gradient)
        
        # 计算波形的衰减率
        peaks, _ = find_peaks(ppg_waveform, height=np.mean(ppg_waveform))
        if len(peaks) >= 2:
            decay_rate = (ppg_waveform[peaks[0]] - ppg_waveform[peaks[-1]]) / len(peaks)
        else:
            decay_rate = 0
        
        # 基于波形特征估算阻力
        base_resistance = 1.0
        
        # 陡峭度越大，阻力越小
        if max_gradient > 0:
            gradient_factor = 1.0 / (1.0 + max_gradient * 0.01)
            base_resistance *= gradient_factor
        
        # 衰减率越大，阻力越大
        if decay_rate > 0:
            decay_factor = 1.0 + decay_rate * 0.1
            base_resistance *= decay_factor
        
        return max(0.1, base_resistance)
    
    def windkessel_model(self, ppg_waveform, patient_params=None):
        """
        基于Windkessel模型估算血流参数
        
        Args:
            ppg_waveform: PPG波形
            patient_params: 患者参数字典 {'age': 年龄, 'bmi': BMI, 'systolic_bp': 收缩压}
            
        Returns:
            flow_params: 血流参数字典
        """
        if len(ppg_waveform) < 100:
            return None
        
        # 提取波形特征
        systolic_peak = np.max(ppg_waveform)
        diastolic_valley = np.min(ppg_waveform)
        pulse_pressure = systolic_peak - diastolic_valley
        
        # 获取患者参数
        if patient_params is None:
            patient_params = {}
        
        age = patient_params.get('age', 50)
        bmi = patient_params.get('bmi', 24)
        systolic_bp = patient_params.get('systolic_bp', 120)
        
        # 血管参数估算
        compliance = self.estimate_compliance(age, bmi, systolic_bp)
        resistance = self.estimate_resistance(ppg_waveform)
        
        # 血流速度估算 (相对值)
        if resistance * compliance > 0:
            flow_velocity_index = pulse_pressure / (resistance * compliance)
        else:
            flow_velocity_index = 0
        
        # 计算其他血流动力学参数
        mean_pressure = np.mean(ppg_waveform)
        cardiac_output_index = flow_velocity_index * mean_pressure
        
        flow_params = {
            'flow_velocity_index': flow_velocity_index,
            'cardiac_output_index': cardiac_output_index,
            'vascular_compliance': compliance,
            'vascular_resistance': resistance,
            'pulse_pressure': pulse_pressure,
            'mean_pressure': mean_pressure
        }
        
        return flow_params
    
    def extract_ppg_features(self, ppg_signal):
        """
        提取PPG多维特征用于机器学习
        
        Args:
            ppg_signal: PPG信号
            
        Returns:
            features: 特征字典
        """
        if len(ppg_signal) < 50:
            return {}
        
        # 基础统计特征
        amplitude = np.max(ppg_signal) - np.min(ppg_signal)
        mean_value = np.mean(ppg_signal)
        std_value = np.std(ppg_signal)
        
        # 波形形态特征
        rise_time = self.calculate_rise_time(ppg_signal)
        area_under_curve = np.trapz(ppg_signal)
        
        # 统计特征
        skewness = scipy.stats.skew(ppg_signal)
        kurtosis = scipy.stats.kurtosis(ppg_signal)
        
        # 频域特征
        spectral_centroid = self.calculate_spectral_centroid(ppg_signal)
        spectral_bandwidth = self.calculate_spectral_bandwidth(ppg_signal)
        
        features = {
            'amplitude': amplitude,
            'mean_value': mean_value,
            'std_value': std_value,
            'rise_time': rise_time,
            'area_under_curve': area_under_curve,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth
        }
        
        return features
    
    def calculate_rise_time(self, ppg_signal):
        """
        计算PPG信号的上升时间
        
        Args:
            ppg_signal: PPG信号
            
        Returns:
            rise_time: 上升时间 (秒)
        """
        peaks, _ = find_peaks(ppg_signal, height=np.mean(ppg_signal))
        valleys, _ = find_peaks(-ppg_signal, height=-np.mean(ppg_signal))
        
        if len(peaks) == 0 or len(valleys) == 0:
            return 0
        
        rise_times = []
        for peak in peaks:
            # 找到最近的谷值
            preceding_valleys = valleys[valleys < peak]
            if len(preceding_valleys) > 0:
                valley = preceding_valleys[-1]
                rise_time = (peak - valley) / self.sampling_rate
                rise_times.append(rise_time)
        
        return np.mean(rise_times) if rise_times else 0
    
    def calculate_spectral_centroid(self, ppg_signal):
        """
        计算频谱质心
        
        Args:
            ppg_signal: PPG信号
            
        Returns:
            spectral_centroid: 频谱质心
        """
        # FFT分析
        fft_values = np.fft.fft(ppg_signal)
        frequencies = np.fft.fftfreq(len(ppg_signal), 1/self.sampling_rate)
        
        # 只取正频率部分
        positive_freq_idx = frequencies > 0
        frequencies = frequencies[positive_freq_idx]
        power_spectrum = np.abs(fft_values[positive_freq_idx])**2
        
        if np.sum(power_spectrum) == 0:
            return 0
        
        # 计算频谱质心
        spectral_centroid = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
        return spectral_centroid
    
    def calculate_spectral_bandwidth(self, ppg_signal):
        """
        计算频谱带宽
        
        Args:
            ppg_signal: PPG信号
            
        Returns:
            spectral_bandwidth: 频谱带宽
        """
        # FFT分析
        fft_values = np.fft.fft(ppg_signal)
        frequencies = np.fft.fftfreq(len(ppg_signal), 1/self.sampling_rate)
        
        # 只取正频率部分
        positive_freq_idx = frequencies > 0
        frequencies = frequencies[positive_freq_idx]
        power_spectrum = np.abs(fft_values[positive_freq_idx])**2
        
        if np.sum(power_spectrum) == 0:
            return 0
        
        # 计算频谱质心
        spectral_centroid = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
        
        # 计算频谱带宽
        spectral_bandwidth = np.sqrt(np.sum(((frequencies - spectral_centroid)**2) * power_spectrum) / np.sum(power_spectrum))
        return spectral_bandwidth
    
    def relative_flow_change(self, ppg_baseline, ppg_current):
        """
        计算相对血流变化
        
        Args:
            ppg_baseline: 基线PPG信号
            ppg_current: 当前PPG信号
            
        Returns:
            flow_change_ratio: 血流变化率 (%)
        """
        pi_baseline, _ = self.calculate_perfusion_index(ppg_baseline)
        pi_current, _ = self.calculate_perfusion_index(ppg_current)
        
        if pi_baseline is None or pi_current is None or pi_baseline == 0:
            return None
        
        flow_change_ratio = (pi_current - pi_baseline) / pi_baseline * 100
        return flow_change_ratio
    
    def reactive_hyperemia_index(self, ppg_pre, ppg_post_occlusion):
        """
        反应性充血指数，评估血管内皮功能
        
        Args:
            ppg_pre: 阻断前PPG信号
            ppg_post_occlusion: 阻断后PPG信号
            
        Returns:
            rhi: 反应性充血指数
            endothelial_function: 内皮功能评估
        """
        pi_pre, _ = self.calculate_perfusion_index(ppg_pre)
        
        if pi_pre is None:
            return None, "数据不足"
        
        # 滑动窗口分析阻断后信号
        window_size = min(len(ppg_post_occlusion) // 10, 500)
        if window_size < 50:
            return None, "数据不足"
        
        pi_values = []
        for i in range(0, len(ppg_post_occlusion) - window_size, window_size // 2):
            segment = ppg_post_occlusion[i:i + window_size]
            pi_segment, _ = self.calculate_perfusion_index(segment)
            if pi_segment is not None:
                pi_values.append(pi_segment)
        
        if not pi_values:
            return None, "分析失败"
        
        pi_peak = max(pi_values)
        rhi = pi_peak / pi_pre
        
        # 内皮功能评估
        if rhi > 1.67:
            endothelial_function = "正常内皮功能"
        elif rhi > 1.2:
            endothelial_function = "轻度内皮功能异常"
        else:
            endothelial_function = "内皮功能异常"
        
        return rhi, endothelial_function
    
    def classify_flow_status(self, pi):
        """
        基于灌注指数分类血流状态
        
        Args:
            pi: 灌注指数
            
        Returns:
            status: 血流状态分类
        """
        if pi is None:
            return "无法评估"
        
        if pi > 5.0:
            return "高灌注"
        elif pi > 0.3:
            return "正常灌注"
        else:
            return "低灌注"
    
    def comprehensive_flow_analysis(self, ppg_signal, patient_params=None):
        """
        综合血流分析
        
        Args:
            ppg_signal: PPG信号
            patient_params: 患者参数
            
        Returns:
            analysis_result: 综合分析结果
        """
        # 灌注指数分析
        pi, flow_status = self.calculate_perfusion_index(ppg_signal)
        
        # Windkessel模型分析
        flow_params = self.windkessel_model(ppg_signal, patient_params)
        
        # 特征提取
        features = self.extract_ppg_features(ppg_signal)
        
        # 组装结果
        analysis_result = {
            'perfusion_analysis': {
                'perfusion_index': pi,
                'flow_status': flow_status
            },
            'hemodynamic_modeling': flow_params,
            'signal_features': features,
            'flow_classification': self.classify_flow_status(pi)
        }
        
        return analysis_result
    
    def estimate_absolute_flow_velocity(self, ppg_signal, calibration_data=None):
        """
        估算绝对血流速度（需要校准数据）
        
        Args:
            ppg_signal: PPG信号
            calibration_data: 校准数据
            
        Returns:
            estimated_velocity: 估算的血流速度 (cm/s)
        """
        # 计算灌注指数
        pi, _ = self.calculate_perfusion_index(ppg_signal)
        
        if pi is None:
            return None
        
        # 基础估算公式（经验性）
        # 这里使用简化的线性关系，实际应用中需要更复杂的模型
        base_velocity = pi * 2.0  # 经验系数
        
        # 如果有校准数据，进行校准
        if calibration_data is not None:
            calibration_factor = calibration_data.get('calibration_factor', 1.0)
            base_velocity *= calibration_factor
        
        # 限制在合理范围内
        estimated_velocity = np.clip(base_velocity, 0.1, 50.0)  # cm/s
        
        return estimated_velocity

def main():
    """测试函数"""
    estimator = PPGBloodFlowEstimator()
    
    # 生成测试数据
    t = np.linspace(0, 10, 1250)  # 10秒，125Hz采样
    test_ppg = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t) + np.random.normal(0, 0.1, len(t))
    
    # 综合分析
    result = estimator.comprehensive_flow_analysis(test_ppg)
    
    print("PPG血流分析结果:")
    print(f"灌注指数: {result['perfusion_analysis']['perfusion_index']:.3f}%")
    print(f"血流状态: {result['perfusion_analysis']['flow_status']}")
    print(f"血流分类: {result['flow_classification']}")

if __name__ == "__main__":
    main()