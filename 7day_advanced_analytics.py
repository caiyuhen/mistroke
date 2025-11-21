#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级7天心梗脑卒中风险评估器
基于PPG信号分析的多因子风险评估模型
专注于心梗和脑卒中特异性特征的精准预测
"""

import json
import os
import glob
import math
import numpy as np
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

class Advanced7DayRiskAssessment:
    """高级7天心梗脑卒中风险评估器"""
    
    def __init__(self):
        """初始化评估器"""
        # 心梗预测模型权重分配
        self.mi_weights = {
            # 心率相关指标：35%
            'resting_hr_elevation': 0.10,      # 静息心率持续升高
            'hr_recovery_delay': 0.10,         # 心率恢复延迟
            'exercise_hr_response': 0.08,      # 运动心率反应异常
            'hr_reserve_decline': 0.07,        # 心率储备下降
            
            # 心律异常：25%
            'pvc_increase': 0.10,              # 室性早搏增多
            'hrv_decline': 0.10,               # 心率变异性急剧下降
            'rhythm_irregularity': 0.05,       # 心律不齐
            
            # 运动功能：20%
            'exercise_tolerance_decline': 0.12, # 运动耐量急剧下降
            'activity_intensity_change': 0.08,  # 活动强度变化
            
            # 自主神经功能：15%
            'autonomic_dysfunction': 0.10,     # HRV指标
            'circadian_rhythm_abnormal': 0.05, # 昼夜节律异常
            
            # 其他指标：5%
            'spo2_decline': 0.05               # 血氧下降
        }
        
        # 脑梗预测模型权重分配
        self.stroke_weights = {
            # 心房颤动相关：40%
            'afib_detection': 0.20,            # 房颤检出
            'pac_frequent': 0.10,              # 房性早搏频发
            'rhythm_irregularity': 0.10,       # 心律不齐增多
            
            # 血管功能：25%
            'bp_circadian_abnormal': 0.12,     # 血压昼夜节律异常
            'vascular_elasticity_decline': 0.08, # 血管弹性下降
            'cerebral_flow_abnormal': 0.05,    # 脑血流异常
            
            # 睡眠呼吸：20%
            'sleep_apnea': 0.12,               # 睡眠呼吸暂停
            'nocturnal_spo2_decline': 0.08,    # 夜间血氧饱和度下降
            
            # 炎症反应：10%
            'temperature_elevation': 0.05,     # 体温轻微升高
            'acute_phase_reaction': 0.05,      # 急性期反应
            
            # 生活方式：5%
            'activity_pattern': 0.03,          # 活动模式
            'stress_response': 0.02            # 应激反应
        }
        
        # 共同风险因素权重
        self.common_risk_weights = {
            'systemic_inflammation': 0.15,      # 系统性炎症
            'hemodynamic_abnormal': 0.15,       # 血液动力学异常
            'autonomic_dysfunction': 0.10,      # 自主神经功能
            'metabolic_dysfunction': 0.10,      # 代谢功能异常
            'lifestyle_factors': 0.10,          # 生活方式因素
            'inflammation_moderate_proportion': 0.08
        }
        
        # 基线风险值（正常人群）
        self.baseline_risks = {
            'mi_7day': 0.0019,      # 7天心梗基线风险
            'mi_30day': 0.0082,     # 30天心梗基线风险
            'stroke_7day': 0.0015,  # 7天脑卒中基线风险
            'stroke_30day': 0.0065  # 30天脑卒中基线风险
        }
        
        # 风险阈值定义
        self.risk_thresholds = {
            'hr_elevation': 85,          # 静息心率 >85 bpm
            'hr_recovery_1min': 12,      # 1分钟心率恢复 <12 bpm
            'target_hr_achievement': 0.85, # 目标心率达成率 <85%
            'hr_reserve': 20,            # 心率储备 <20 bpm
            'pvc_percentage': 0.03,      # 室性早搏 >3%
            'hrv_sdnn': 30,             # SDNN <30ms
            'qtc_prolongation': 450,     # QTc >450ms
            'exercise_decline': 0.30,    # 运动耐量下降 >30%
            'circadian_difference': 0.10, # 昼夜心率差异 <10%
            'spo2_threshold': 95,        # SpO2 <95%
            'afib_duration': 30,         # 房颤 >30秒
            'pac_percentage': 0.05,      # 房性早搏 >5%
            'irregular_rhythm': 0.10,    # 不规则心律 >10%时间
            'p_wave_duration': 120,      # P波时间 >120ms
            'pulse_pressure': 60,        # 脉压差 >60mmHg
            'ahi_threshold': 15,         # AHI >15次/小时
            'min_spo2': 88,             # 最低SpO2 <88%
            'temperature_elevation': 37.2, # 基础体温 >37.2°C
            'bp_variability': 0.15,      # 血压变异系数 >15%
            'sedentary_hours': 10,       # 久坐 >10小时/天
            'sleep_efficiency': 0.75,    # 睡眠效率 <75%
            'stress_index': 120,         # 压力指数 >120
            'pwv_moderate': 10.0,
            'pwv_high': 12.0,
            'vascular_age_moderate': 70,
            'vascular_age_high': 80,
            'inflammation_score_moderate': 3.0,
            'inflammation_score_high': 4.0,
            'inflammation_moderate_pct_threshold': 50,
            'bp_cv_moderate': 15.0,
            'bp_cv_high': 20.0
        }

    def extract_mi_features(self, data: Dict) -> Dict[str, float]:
        """提取心梗相关特征"""
        features = {}
        
        try:
            # 心率相关指标
            hr_stats = data.get('vascular_function_statistics', {}).get('heart_rate_stats', {})
            mean_hr = hr_stats.get('mean', 70)
            
            # 静息心率持续升高
            features['resting_hr_elevation'] = min(1.0, max(0.0, 
                (mean_hr - self.risk_thresholds['hr_elevation']) / 20))
            
            # 心率变异性
            hrv_stats = data.get('inflammation_statistics', {}).get('hrv_statistics', {})
            sdnn = hrv_stats.get('average_sdnn', 50)
            rmssd = hrv_stats.get('average_rmssd', 30)
            
            # HRV急剧下降
            features['hrv_decline'] = max(0.0, min(1.0, 
                (self.risk_thresholds['hrv_sdnn'] - sdnn) / self.risk_thresholds['hrv_sdnn']))
            
            # 心律异常检测
            arrhythmia_stats = data.get('arrhythmia_statistics', {}).get('arrhythmia_detection_statistics', {})
            pvc_rate = arrhythmia_stats.get('pvc_detection_rate', 0)
            normal_rhythm_rate = arrhythmia_stats.get('normal_rhythm_rate', 1.0)
            
            # 室性早搏增多
            features['pvc_increase'] = min(1.0, pvc_rate / self.risk_thresholds['pvc_percentage'])
            
            # 心律不齐
            features['rhythm_irregularity'] = max(0.0, 1.0 - normal_rhythm_rate)
            
            # 运动功能相关
            # 基于血流灌注状态评估运动耐量
            flow_stats = data.get('blood_flow_statistics', {}).get('flow_status_distribution', {})
            low_perfusion = flow_stats.get('低灌注', 0)
            total_segments = data.get('blood_flow_statistics', {}).get('total_analyzed_segments', 1)
            
            # 运动耐量下降（基于低灌注比例）
            features['exercise_tolerance_decline'] = min(1.0, low_perfusion / total_segments)
            
            # 血氧相关
            sleep_analysis = data.get('sleep_analysis', {})
            spo2_analysis = sleep_analysis.get('nocturnal_spo2_analysis', {})
            min_spo2 = spo2_analysis.get('min_spo2', 98)
            
            # 血氧下降
            features['spo2_decline'] = max(0.0, min(1.0, 
                (self.risk_thresholds['spo2_threshold'] - min_spo2) / 10))
            
            # 其他特征默认值
            for key in self.mi_weights.keys():
                if key not in features:
                    features[key] = 0.0
                    
        except Exception as e:
            print(f"提取心梗特征时出错: {e}")
            # 设置默认值
            for key in self.mi_weights.keys():
                features[key] = 0.0
        
        return features

    def extract_stroke_features(self, data: Dict) -> Dict[str, float]:
        """提取脑卒中相关特征"""
        features = {}
        
        try:
            # 心房颤动相关
            arrhythmia_stats = data.get('arrhythmia_statistics', {}).get('arrhythmia_detection_statistics', {})
            afib_rate = arrhythmia_stats.get('afib_detection_rate', 0)
            pac_rate = arrhythmia_stats.get('pac_detection_rate', 0)
            normal_rhythm_rate = arrhythmia_stats.get('normal_rhythm_rate', 1.0)
            
            # 房颤检出
            features['afib_detection'] = min(1.0, afib_rate * 10)  # 放大房颤风险
            
            # 房性早搏频发
            features['pac_frequent'] = min(1.0, pac_rate / self.risk_thresholds['pac_percentage'])
            
            # 心律不齐增多
            features['rhythm_irregularity'] = max(0.0, 1.0 - normal_rhythm_rate)
            
            # 血管功能相关
            sleep_analysis = data.get('sleep_analysis', {})
            bp_rhythm = sleep_analysis.get('blood_pressure_rhythm_analysis', {})
            
            # 血压昼夜节律异常
            dipping_pattern = bp_rhythm.get('nocturnal_dipping', {}).get('dipping_pattern', 'normal')
            if dipping_pattern in ['non_dipper', 'reverse_dipper']:
                features['bp_circadian_abnormal'] = 1.0
            else:
                features['bp_circadian_abnormal'] = 0.0
            features['reverse_dipper_pattern'] = 1.0 if dipping_pattern == 'reverse_dipper' else 0.0
            
            # 血管弹性下降
            vascular_stats = data.get('vascular_function_statistics', {})
            pwv_mean = vascular_stats.get('pwv_stats', {}).get('mean', 8)
            vascular_age_mean = vascular_stats.get('vascular_age_stats', {}).get('mean', 40)
            
            # 基于PWV和血管年龄评估血管弹性
            features['vascular_elasticity_decline'] = min(1.0, max(0.0, 
                (pwv_mean - 12) / 8 + (vascular_age_mean - 70) / 30) / 2)
            pm = self.risk_thresholds.get('pwv_moderate', 10.0)
            ph = self.risk_thresholds.get('pwv_high', 12.0)
            features['pwv_elevation'] = min(1.0, max(0.0, (pwv_mean - pm) / (ph - pm)))
            
            # 睡眠呼吸相关
            sleep_apnea_analysis = sleep_analysis.get('sleep_apnea_analysis', {})
            apnea_events = sleep_apnea_analysis.get('apnea_events', [])
            
            # 睡眠呼吸暂停（基于事件数量和严重程度）
            severe_events = sum(1 for event in apnea_events if event.get('severity') in ['moderate', 'severe'])
            total_events = len(apnea_events)
            
            if total_events > 0:
                features['sleep_apnea'] = min(1.0, (total_events + severe_events * 2) / 20)
            else:
                features['sleep_apnea'] = 0.0
            
            # 夜间血氧饱和度下降
            spo2_analysis = sleep_analysis.get('nocturnal_spo2_analysis', {})
            min_spo2 = spo2_analysis.get('min_spo2', 98)
            spo2_below_90 = spo2_analysis.get('spo2_below_90_percent', 0)
            
            features['nocturnal_spo2_decline'] = max(0.0, min(1.0, 
                (self.risk_thresholds['min_spo2'] - min_spo2) / 10 + spo2_below_90))
            
            # 炎症反应
            inflammation_stats = data.get('inflammation_statistics', {})
            avg_inflammation_score = inflammation_stats.get('average_inflammation_score', 2)
            
            # 体温轻微升高（基于炎症评分）
            features['temperature_elevation'] = min(1.0, max(0.0, (avg_inflammation_score - 3) / 2))
            
            # 其他特征默认值
            for key in self.stroke_weights.keys():
                if key not in features:
                    features[key] = 0.0
                    
        except Exception as e:
            print(f"提取脑卒中特征时出错: {e}")
            # 设置默认值
            for key in self.stroke_weights.keys():
                features[key] = 0.0
        
        return features

    def extract_common_risk_features(self, data: Dict) -> Dict[str, float]:
        """提取共同风险因素特征"""
        features = {}
        
        try:
            # 系统性炎症
            inflammation_stats = data.get('inflammation_statistics', {})
            avg_inflammation_score = inflammation_stats.get('average_inflammation_score', 2)
            features['systemic_inflammation'] = min(1.0, max(0.0, (avg_inflammation_score - 2) / 3))
            
            # 血液动力学异常
            sleep_analysis = data.get('sleep_analysis', {})
            bp_rhythm = sleep_analysis.get('blood_pressure_rhythm_analysis', {})
            bp_variability = bp_rhythm.get('bp_variability', {}).get('coefficient_of_variation', 10)
            
            features['hemodynamic_abnormal'] = min(1.0, max(0.0, 
                (bp_variability - self.risk_thresholds['bp_variability'] * 100) / 20))
            cv = bp_variability
            cv_m = self.risk_thresholds.get('bp_cv_moderate', 15.0)
            cv_h = self.risk_thresholds.get('bp_cv_high', 20.0)
            features['bp_variability_moderate'] = 0.5 if cv >= cv_m and cv < cv_h else (0.9 if cv >= cv_h else 0.0)
            
            # 自主神经功能
            hrv_stats = inflammation_stats.get('hrv_statistics', {})
            lf_hf_ratio = hrv_stats.get('average_lf_hf_ratio', 1.0)
            
            # 基于LF/HF比值评估自主神经功能
            features['autonomic_dysfunction'] = min(1.0, max(0.0, abs(lf_hf_ratio - 1.0)))
            
            # 代谢功能异常（基于血管年龄和炎症）
            vascular_stats = data.get('vascular_function_statistics', {})
            vascular_age_mean = vascular_stats.get('vascular_age_stats', {}).get('mean', 40)
            
            features['metabolic_dysfunction'] = min(1.0, max(0.0, (vascular_age_mean - 60) / 30))
            vm = self.risk_thresholds.get('vascular_age_moderate', 70)
            vh = self.risk_thresholds.get('vascular_age_high', 80)
            features['vascular_age_moderate'] = min(1.0, max(0.0, (vascular_age_mean - vm) / (vh - vm)))
            
            # 生活方式因素（基于睡眠质量和活动水平）
            sleep_quality = sleep_analysis.get('nocturnal_spo2_analysis', {}).get('analysis_quality', 'good')
            quality_score = {'excellent': 0, 'good': 0.2, 'fair': 0.5, 'poor': 0.8}.get(sleep_quality, 0.5)
            
            features['lifestyle_factors'] = quality_score
            is_m = self.risk_thresholds.get('inflammation_score_moderate', 3.0)
            is_h = self.risk_thresholds.get('inflammation_score_high', 4.0)
            features['inflammation_score_moderate'] = min(1.0, max(0.0, (avg_inflammation_score - is_m) / (is_h - is_m)))
            moderate_pct = inflammation_stats.get('inflammation_grade_distribution', {}).get('moderate', {}).get('percentage', 0)
            threshold_pct = self.risk_thresholds.get('inflammation_moderate_pct_threshold', 50)
            features['inflammation_moderate_proportion'] = 0.8 if moderate_pct >= threshold_pct else 0.0
            
        except Exception as e:
            print(f"提取共同风险因素时出错: {e}")
            # 设置默认值
            for key in self.common_risk_weights.keys():
                features[key] = 0.0
        
        return features

    def calculate_risk_score(self, features: Dict[str, float], weights: Dict[str, float]) -> float:
        """计算风险评分"""
        total_score = 0.0
        total_weight = 0.0
        
        for feature, weight in weights.items():
            if feature in features:
                total_score += features[feature] * weight
                total_weight += weight
        
        # 归一化到0-1范围
        if total_weight > 0:
            return min(1.0, total_score / total_weight)
        return 0.0

    def calculate_risk_ratio(self, risk_score: float, baseline_risk: float, days: int = 7) -> float:
        """计算风险比率"""
        # 使用指数函数计算风险比率
        risk_multiplier = math.exp(risk_score * 3.0)  # 调节系数
        return baseline_risk * risk_multiplier

    def get_risk_level(self, risk_score: float, risk_type: str) -> str:
        """获取风险等级"""
        if risk_type == "mi":
            if risk_score < 0.14:
                return "低风险"
            elif risk_score < 0.17:  # 0.26-0.30 为中风险
                return "中风险"
            else:
                return "高风险"
        elif risk_type == "stroke":
            if risk_score < 0.28:
                return "低风险"
            elif risk_score < 0.31:  # 0.28-0.30 为中风险
                return "中风险"
            else:
                return "高风险"
        else:
            return "未知类型"

    def generate_risk_factors_analysis(self, mi_features: Dict, stroke_features: Dict, 
                                     common_features: Dict) -> Dict:
        """生成风险因素分析"""
        primary_factors = []
        secondary_factors = []
        protective_factors = []
        
        # 分析心梗风险因素
        for feature, value in mi_features.items():
            if value > 0.7:
                if feature == 'resting_hr_elevation':
                    primary_factors.append("静息心率持续升高")
                elif feature == 'hrv_decline':
                    primary_factors.append("心率变异性急剧下降")
                elif feature == 'pvc_increase':
                    primary_factors.append("室性早搏显著增多")
                elif feature == 'exercise_tolerance_decline':
                    primary_factors.append("运动耐量急剧下降")
            elif value > 0.4:
                if feature == 'rhythm_irregularity':
                    secondary_factors.append("心律不齐增多")
                elif feature == 'spo2_decline':
                    secondary_factors.append("血氧饱和度下降")
        
        # 分析脑卒中风险因素
        for feature, value in stroke_features.items():
            if value > 0.7:
                if feature == 'afib_detection':
                    primary_factors.append("房颤检出")
                elif feature == 'bp_circadian_abnormal':
                    primary_factors.append("血压昼夜节律异常")
                elif feature == 'sleep_apnea':
                    primary_factors.append("睡眠呼吸暂停")
                elif feature == 'vascular_elasticity_decline':
                    primary_factors.append("血管弹性显著下降")
                elif feature == 'pwv_elevation':
                    primary_factors.append("PWV显著升高")
                elif feature == 'reverse_dipper_pattern':
                    primary_factors.append("反杓型血压模式")
            elif value > 0.4:
                if feature == 'pac_frequent':
                    secondary_factors.append("房性早搏频发")
                elif feature == 'nocturnal_spo2_decline':
                    secondary_factors.append("夜间血氧饱和度下降")
        
        # 分析共同风险因素
        for feature, value in common_features.items():
            if value > 0.6:
                if feature == 'systemic_inflammation':
                    primary_factors.append("系统性炎症反应")
                elif feature == 'hemodynamic_abnormal':
                    primary_factors.append("血液动力学异常")
                elif feature == 'inflammation_moderate_proportion':
                    primary_factors.append("中度炎症比例高")
            elif value > 0.3:
                if feature == 'autonomic_dysfunction':
                    secondary_factors.append("自主神经功能异常")
                elif feature == 'metabolic_dysfunction':
                    secondary_factors.append("代谢功能异常")
                elif feature == 'vascular_age_moderate':
                    secondary_factors.append("血管年龄轻度老化")
                elif feature == 'inflammation_score_moderate':
                    secondary_factors.append("炎症评分中等")
                elif feature == 'bp_variability_moderate':
                    secondary_factors.append("血压变异性轻度增高")
        
        # 保护性因素（基于正常指标）
        if mi_features.get('rhythm_irregularity', 1) < 0.2:
            protective_factors.append("心律基本正常")
        if stroke_features.get('afib_detection', 1) < 0.1:
            protective_factors.append("无房颤检出")
        if common_features.get('systemic_inflammation', 1) < 0.3:
            protective_factors.append("炎症水平较低")
        
        return {
            'primary_risk_factors': primary_factors[:5],  # 最多5个主要因素
            'secondary_risk_factors': secondary_factors[:5],  # 最多5个次要因素
            'protective_factors': protective_factors[:3]  # 最多3个保护因素
        }

    def generate_recommendations(self, mi_risk_level: str, stroke_risk_level: str, 
                               risk_factors: Dict) -> List[str]:
        """生成个性化建议"""
        recommendations = []
        
        # 基于风险等级的基础建议
        if mi_risk_level in ["高风险", "极高风险"] or stroke_risk_level in ["高风险", "极高风险"]:
            recommendations.extend([
                "立即就医进行专业心脑血管评估",
                "48小时内完成心电图、超声心动图检查",
                "考虑进行冠状动脉造影或颈动脉超声检查"
            ])
        elif mi_risk_level == "中风险" or stroke_risk_level == "中风险":
            recommendations.extend([
                "建议1-2周内就医进行心血管检查",
                "定期监测血压、心率变化"
            ])
        
        # 基于具体风险因素的建议
        primary_factors = risk_factors.get('primary_risk_factors', [])
        
        if any("心率" in factor for factor in primary_factors):
            recommendations.append("严格控制心率，避免剧烈运动")
        
        if any("房颤" in factor for factor in primary_factors):
            recommendations.append("立即进行抗凝治疗评估")
        
        if any("血压" in factor for factor in primary_factors):
            recommendations.append("24小时血压监测，调整降压方案")
        
        if any("睡眠" in factor for factor in primary_factors):
            recommendations.append("进行睡眠呼吸监测，考虑CPAP治疗")
        
        if any("炎症" in factor for factor in primary_factors):
            recommendations.append("检查炎症标志物，必要时抗炎治疗")
        
        # 通用建议
        recommendations.extend([
            "严格控制血压、血脂和血糖",
            "戒烟限酒，保持适当体重",
            "规律有氧运动，每周至少150分钟",
            "保持充足睡眠，管理压力",
            "定期使用PPG设备进行健康监测"
        ])
        
        return list(set(recommendations))  # 去重

    def get_30day_risk_level(self, risk_ratio: float, baseline_risk: float) -> str:
        """获取30天风险等级（基于风险比率）"""
        # 计算相对于基线风险的倍数
        risk_multiplier = risk_ratio / baseline_risk
        
        if risk_multiplier < 2.0:
            return "低风险"
        elif risk_multiplier < 5.0:
            return "中风险"
        elif risk_multiplier < 10.0:
            return "高风险"
        else:
            return "极高风险"

    def assess_risk(self, data: Dict) -> Dict:
        """评估风险"""
        try:
            # 提取特征
            mi_features = self.extract_mi_features(data)
            stroke_features = self.extract_stroke_features(data)
            common_features = self.extract_common_risk_features(data)
            
            # 计算风险评分
            mi_risk_score = self.calculate_risk_score(mi_features, self.mi_weights)
            stroke_risk_score = self.calculate_risk_score(stroke_features, self.stroke_weights)
            
            # 加入共同风险因素
            common_risk_score = self.calculate_risk_score(common_features, self.common_risk_weights)
            
            # 综合风险评分
            final_mi_score = min(1.0, mi_risk_score + common_risk_score * 0.3)
            final_stroke_score = min(1.0, stroke_risk_score + common_risk_score * 0.3)
            
            # 计算风险比率和倍数
            mi_7day_ratio = self.calculate_risk_ratio(final_mi_score, self.baseline_risks['mi_7day'])
            mi_30day_ratio = self.calculate_risk_ratio(final_mi_score, self.baseline_risks['mi_30day'])
            stroke_7day_ratio = self.calculate_risk_ratio(final_stroke_score, self.baseline_risks['stroke_7day'])
            stroke_30day_ratio = self.calculate_risk_ratio(final_stroke_score, self.baseline_risks['stroke_30day'])
            
            # 计算风险倍数
            mi_7day_multiplier = mi_7day_ratio / self.baseline_risks['mi_7day']
            mi_30day_multiplier = mi_30day_ratio / self.baseline_risks['mi_30day']
            stroke_7day_multiplier = stroke_7day_ratio / self.baseline_risks['stroke_7day']
            stroke_30day_multiplier = stroke_30day_ratio / self.baseline_risks['stroke_30day']
            
            # 获取风险等级
            mi_risk_level = self.get_risk_level(final_mi_score, "mi")
            stroke_risk_level = self.get_risk_level(final_stroke_score, "stroke")
            
            # 获取30天风险等级
            mi_30day_risk_level = self.get_30day_risk_level(mi_30day_ratio, self.baseline_risks['mi_30day'])
            stroke_30day_risk_level = self.get_30day_risk_level(stroke_30day_ratio, self.baseline_risks['stroke_30day'])
            
            # 生成风险因素分析
            risk_factors_analysis = self.generate_risk_factors_analysis(
                mi_features, stroke_features, common_features)
            
            # 生成建议
            recommendations = self.generate_recommendations(
                mi_risk_level, stroke_risk_level, risk_factors_analysis)
            
            return {
                'device_id': data.get('device_id', 'unknown'),
                'collect_time': data.get('detailed_vascular_analysis', [{}])[0].get('collect_time'),
                'analysis_timestamp': datetime.now().isoformat(),
                'risk_prediction': {
                    'myocardial_infarction': {
                        '7_day_risk_ratio': mi_7day_ratio,
                        '30_day_risk_ratio': mi_30day_ratio,
                        'risk_score': final_mi_score,
                        'risk_level': mi_risk_level,
                        '7_day_percentage': mi_7day_ratio * 100,
                        '30_day_percentage': mi_30day_ratio * 100,
                        '7_day_risk_level': self.get_risk_level(final_mi_score, "mi"),
                        '30_day_risk_level': mi_30day_risk_level,
                        '7_day_multiplier': mi_7day_multiplier,
                        '30_day_multiplier': mi_30day_multiplier
                    },
                    'stroke': {
                        '7_day_risk_ratio': stroke_7day_ratio,
                        '30_day_risk_ratio': stroke_30day_ratio,
                        'risk_score': final_stroke_score,
                        'risk_level': stroke_risk_level,
                        '7_day_percentage': stroke_7day_ratio * 100,
                        '30_day_percentage': stroke_30day_ratio * 100,
                        '7_day_risk_level': self.get_risk_level(final_stroke_score, "stroke"),
                        '30_day_risk_level': stroke_30day_risk_level,
                        '7_day_multiplier': stroke_7day_multiplier,
                        '30_day_multiplier': stroke_30day_multiplier
                    }
                },
                'feature_analysis': {
                    'mi_features': mi_features,
                    'stroke_features': stroke_features,
                    'common_features': common_features
                },
                'risk_factors_analysis': risk_factors_analysis,
                'recommendations': recommendations,
                'calculation_details': {
                    'baseline_risks': self.baseline_risks,
                    'risk_weights': {
                        'mi_weights': self.mi_weights,
                        'stroke_weights': self.stroke_weights,
                        'common_weights': self.common_risk_weights
                    },
                    'methodology': "基于PPG信号分析的高级多因子风险评估模型v2.0"
                }
            }
            
        except Exception as e:
            print(f"风险评估过程中出错: {e}")
            return {
                'device_id': data.get('device_id', 'unknown'),
                'analysis_timestamp': datetime.now().isoformat(),
                'error': str(e),
                'risk_prediction': {
                    'myocardial_infarction': {
                        '7_day_risk_ratio': 0.0,
                        '30_day_risk_ratio': 0.0,
                        'risk_score': 0.0,
                        'risk_level': "无法评估",
                        '7_day_percentage': 0.0,
                        '30_day_percentage': 0.0,
                        '7_day_risk_level': "无法评估",
                        '7_day_multiplier': 1.0,
                        '30_day_multiplier': 1.0
                    },
                    'stroke': {
                        '7_day_risk_ratio': 0.0,
                        '30_day_risk_ratio': 0.0,
                        'risk_score': 0.0,
                        'risk_level': "无法评估",
                        '7_day_percentage': 0.0,
                        '30_day_percentage': 0.0,
                        '7_day_risk_level': "无法评估",
                        '7_day_multiplier': 1.0,
                        '30_day_multiplier': 1.0
                    }
                }
            }

    def process_single_file(self, input_file: str, output_dir: str) -> Dict:
        """处理单个文件"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 进行风险评估
            result = self.assess_risk(data)
            
            # 生成输出文件名 - 从输入文件名提取时间戳信息
            input_basename = os.path.basename(input_file)
            device_id = result['device_id']
            
            # 尝试从输入文件名中提取时间戳
            if '_vascular_analysis_' in input_basename:
                # 提取时间戳部分，例如从 "108932503058773_vascular_analysis_20251031_140348.json" 提取 "20251031_140348"
                timestamp_part = input_basename.split('_vascular_analysis_')[1].replace('.json', '')
                output_filename = f"{device_id}_advanced_7day_risk_assessment_{timestamp_part}.json"
            else:
                # 如果无法提取时间戳，使用当前时间
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{device_id}_advanced_7day_risk_assessment_{current_time}.json"
            
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存结果
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            return {
                'input_file': input_file,
                'output_file': output_path,
                'device_id': device_id,
                'collect_time': result.get('collect_time'),
                'mi_risk_level': result['risk_prediction']['myocardial_infarction']['risk_level'],
                'stroke_risk_level': result['risk_prediction']['stroke']['risk_level'],
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'input_file': input_file,
                'error': str(e),
                'status': 'failed'
            }

    def batch_process_files(self, input_dir: str, output_dir: str) -> Dict:
        """批量处理文件"""
        # 将相对路径转换为绝对路径
        input_dir = os.path.abspath(input_dir)
        output_dir = os.path.abspath(output_dir)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有JSON文件
        json_files = glob.glob(os.path.join(input_dir, "*.json"))
        
        if not json_files:
            return {
                'total_files': 0,
                'processed_files': 0,
                'failed_files': 0,
                'results': [],
                'message': f"在目录 {input_dir} 中未找到JSON文件"
            }
        
        results = []
        processed_count = 0
        failed_count = 0
        
        print(f"开始处理 {len(json_files)} 个文件...")
        
        for i, json_file in enumerate(json_files, 1):
            print(f"处理文件 {i}/{len(json_files)}: {os.path.basename(json_file)}")
            
            result = self.process_single_file(json_file, output_dir)
            results.append(result)
            
            if result['status'] == 'success':
                processed_count += 1
                print(f"  ✓ 成功 - 心梗风险: {result['mi_risk_level']}, 脑卒中风险: {result['stroke_risk_level']}")
            else:
                failed_count += 1
                print(f"  ✗ 失败 - {result.get('error', '未知错误')}")
        
        # 生成批量处理摘要
        summary = {
            'total_files': len(json_files),
            'processed_files': processed_count,
            'failed_files': failed_count,
            'processing_timestamp': datetime.now().isoformat(),
            'results': [{'device_id': r['device_id'], 'collect_time': r.get('collect_time'), 'mi_risk_level': r['mi_risk_level'], 'stroke_risk_level': r['stroke_risk_level'], 'status': r['status']} for r in results if r['status'] == 'success']
        }
        
        # 保存摘要文件
        summary_path = os.path.join(output_dir, "advanced_7day_risk_batch_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n批量处理完成!")
        print(f"总文件数: {len(json_files)}")
        print(f"成功处理: {processed_count}")
        print(f"处理失败: {failed_count}")
        print(f"摘要文件: {summary_path}")
        
        return summary

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='高级7天心梗脑卒中风险评估器 - 支持相对路径输入')
    parser.add_argument('-i', '--input', 
                       default='./analysis_results',
                       help='输入分析目录路径（支持相对路径，默认: ./analysis_results）')
    parser.add_argument('-o', '--output', 
                       default='./risk_results',
                       help='输出结果目录路径（支持相对路径，默认: ./risk_results）')
    parser.add_argument('-v', '--verbose', 
                       action='store_true',
                       help='显示详细处理信息')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建评估器实例
    assessor = Advanced7DayRiskAssessment()
    
    # 使用命令行参数或默认值
    input_dir = args.input
    output_dir = args.output
    
    # 显示使用的路径信息
    abs_input_dir = os.path.abspath(input_dir)
    abs_output_dir = os.path.abspath(output_dir)
    
    print(f"\n=== 高级7天心梗脑卒中风险评估器 ===")
    print(f"输入目录: {input_dir} -> {abs_input_dir}")
    print(f"输出目录: {output_dir} -> {abs_output_dir}")
    
    # 检查输入目录是否存在
    if not os.path.exists(abs_input_dir):
        print(f"错误: 输入目录不存在: {abs_input_dir}")
        return
    
    # 批量处理文件
    summary = assessor.batch_process_files(input_dir, output_dir)
    
    print(f"\n=== 处理完成 ===")
    print(f"总文件数: {summary.get('total_files', 0)}")
    print(f"成功处理: {summary.get('processed_files', 0)}")
    print(f"处理失败: {summary.get('failed_files', 0)}")
    
    if 'message' in summary:
        print(f"消息: {summary['message']}")
    
    print(f"\n结果已保存到: {abs_output_dir}")
    
    return summary

if __name__ == "__main__":
    main()