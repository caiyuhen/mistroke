#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
血管功能分析处理脚本
处理JSON文件中的PPG数据，进行血管功能分析和血流分析
"""

import os
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import multiprocessing
from datetime import datetime, timedelta
from typing import Dict, Any
from ppg_vascular_analysis import PPGVascularAnalyzer
from ppg_blood_flow_analysis import PPGBloodFlowEstimator
from ppg_arrhythmia_detection import PPGArrhythmiaDetector
from ppg_inflammation_detection import PPGInflammationDetector
from ppg_sleep_analysis import analyze_sleep_data
import traceback

# 优先使用 orjson 进行更快的 JSON 解析/序列化，失败则回退到标准库
try:
    import orjson as _orjson
except Exception:
    _orjson = None


def _process_and_save_entry(file_path: str, input_dir: str, output_dir: str):
    """进程池工作入口：加载并处理单个文件并保存结果。

    为避免主进程中的对象跨进程序列化开销，在子进程内创建处理器。
    返回 (success, filename)。
    """
    try:
        processor = VascularAnalysisProcessor(input_dir=input_dir, output_dir=output_dir)
        result = processor.process_device_data(file_path)
        if result is not None:
            processor.save_analysis_result(result['device_id'], result)
            return True, os.path.basename(file_path)
        return False, os.path.basename(file_path)
    except Exception:
        return False, os.path.basename(file_path)

class VascularAnalysisProcessor:
    def __init__(self, input_dir="output", output_dir="analysis_results"):
        """
        初始化处理器
        
        Args:
            input_dir: 输入目录，包含JSON文件
            output_dir: 输出目录，保存分析结果
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.analyzer = PPGVascularAnalyzer()
        self.blood_flow_estimator = PPGBloodFlowEstimator()  # 血流分析器
        self.arrhythmia_detector = PPGArrhythmiaDetector()  # 心律不齐检测器
        self.inflammation_detector = PPGInflammationDetector()  # 炎症检测器
        self._offset_cache = {}
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_json_file(self, file_path):
        """
        加载JSON文件
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            data: JSON数据
        """
        try:
            if _orjson is not None:
                with open(file_path, 'rb') as f:
                    data = _orjson.loads(f.read())
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            return data
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            return None

    def _get_offsets(self, n: int):
        k = int(n)
        v = self._offset_cache.get(k)
        if v is None:
            v = np.arange(k, dtype='timedelta64[ms]') * 8
            self._offset_cache[k] = v
        return v
    
    def calculate_arrhythmia_statistics(self, arrhythmia_results):
        """
        计算心律不齐统计信息
        
        Args:
            arrhythmia_results: 心律不齐分析结果列表
            
        Returns:
            statistics: 统计信息字典
        """
        if not arrhythmia_results:
            return {}
        
        # 提取各项指标
        mean_ppi_values = []
        sdnn_values = []
        rmssd_values = []
        pnn50_values = []
        cv_values = []
        irregularity_values = []
        lf_hf_ratios = []
        
        # 心律不齐检测结果统计
        afib_detections = []
        pac_detections = []
        pvc_detections = []
        sinus_arrhythmia_detections = []
        
        for result in arrhythmia_results:
            # 时域特征
            time_features = result.get('time_domain_features', {})
            if time_features.get('mean_ppi') is not None:
                mean_ppi_values.append(time_features['mean_ppi'])
            if time_features.get('sdnn') is not None:
                sdnn_values.append(time_features['sdnn'])
            if time_features.get('rmssd') is not None:
                rmssd_values.append(time_features['rmssd'])
            if time_features.get('pnn50') is not None:
                pnn50_values.append(time_features['pnn50'])
            if time_features.get('cv') is not None:
                cv_values.append(time_features['cv'])
            if time_features.get('irregularity_index') is not None:
                irregularity_values.append(time_features['irregularity_index'])
            
            # 频域特征
            freq_features = result.get('frequency_domain_features', {})
            if freq_features.get('lf_hf_ratio') is not None:
                lf_hf_ratios.append(freq_features['lf_hf_ratio'])
            
            # 心律不齐检测结果
            detection = result.get('arrhythmia_detection', {})
            afib_detections.append(detection.get('atrial_fibrillation', {}).get('detected', False))
            pac_detections.append(detection.get('premature_beats', {}).get('pac_detected', False))
            pvc_detections.append(detection.get('premature_beats', {}).get('pvc_detected', False))
            sinus_arrhythmia_detections.append(detection.get('sinus_arrhythmia', {}).get('detected', False))
        
        # 计算统计值
        statistics = {
            'time_domain_statistics': {
                'mean_ppi': self._calculate_stats(mean_ppi_values),
                'sdnn': self._calculate_stats(sdnn_values),
                'rmssd': self._calculate_stats(rmssd_values),
                'pnn50': self._calculate_stats(pnn50_values),
                'cv': self._calculate_stats(cv_values),
                'irregularity_index': self._calculate_stats(irregularity_values)
            },
            'frequency_domain_statistics': {
                'lf_hf_ratio': self._calculate_stats(lf_hf_ratios)
            },
            'arrhythmia_detection_statistics': {
                'total_segments': len(arrhythmia_results),
                'afib_detection_rate': sum(afib_detections) / len(afib_detections) if afib_detections else 0,
                'pac_detection_rate': sum(pac_detections) / len(pac_detections) if pac_detections else 0,
                'pvc_detection_rate': sum(pvc_detections) / len(pvc_detections) if pvc_detections else 0,
                'sinus_arrhythmia_rate': sum(sinus_arrhythmia_detections) / len(sinus_arrhythmia_detections) if sinus_arrhythmia_detections else 0,
                'normal_rhythm_rate': 1 - max(sum(afib_detections), sum(pac_detections), sum(pvc_detections)) / len(arrhythmia_results) if arrhythmia_results else 0
            }
        }
        
        return statistics
    
    def assess_arrhythmia_risk(self, arrhythmia_statistics):
        """
        评估心律不齐风险
        
        Args:
            arrhythmia_statistics: 心律不齐统计信息
            
        Returns:
            risk_assessment: 风险评估结果
        """
        if not arrhythmia_statistics:
            return {
                'risk_level': 'unknown',
                'risk_score': 0,
                'recommendations': ['数据不足，无法评估心律不齐风险']
            }
        
        risk_score = 0
        risk_factors = []
        recommendations = []
        
        # 获取检测统计
        detection_stats = arrhythmia_statistics.get('arrhythmia_detection_statistics', {})
        time_stats = arrhythmia_statistics.get('time_domain_statistics', {})
        
        # 房颤风险评估
        afib_rate = detection_stats.get('afib_detection_rate', 0)
        if afib_rate > 0.1:  # 超过10%的数据段检测到房颤
            risk_score += 30
            risk_factors.append('检测到房颤迹象')
            recommendations.append('建议进行心电图检查确认房颤诊断')
        
        # 早搏风险评估
        pac_rate = detection_stats.get('pac_detection_rate', 0)
        pvc_rate = detection_stats.get('pvc_detection_rate', 0)
        if pac_rate > 0.05 or pvc_rate > 0.05:  # 超过5%的数据段检测到早搏
            risk_score += 15
            risk_factors.append('检测到频繁早搏')
            recommendations.append('建议监测心律变化，必要时就医')
        
        # 心率变异性评估
        cv_stats = time_stats.get('cv', {})
        cv_mean = cv_stats.get('mean') if cv_stats else None
        if cv_mean is not None and cv_mean > 20:  # CV > 20%
            risk_score += 20
            risk_factors.append('心率变异性异常增高')
            recommendations.append('建议评估自主神经功能')
        
        # RMSSD评估
        rmssd_stats = time_stats.get('rmssd', {})
        rmssd_mean = rmssd_stats.get('mean') if rmssd_stats else None
        if rmssd_mean is not None and rmssd_mean > 50:  # RMSSD > 50ms
            risk_score += 10
            risk_factors.append('短期心率变异性增高')
        
        # 确定风险等级
        if risk_score >= 50:
            risk_level = 'high'
            risk_description = '高风险'
        elif risk_score >= 25:
            risk_level = 'moderate'
            risk_description = '中等风险'
        elif risk_score >= 10:
            risk_level = 'low'
            risk_description = '低风险'
        else:
            risk_level = 'normal'
            risk_description = '正常'
        
        # 基础建议
        if risk_level == 'normal':
            recommendations.append('心律状况良好，建议保持健康生活方式')
        else:
            recommendations.append('建议定期监测心律变化')
            recommendations.append('如有胸闷、心悸症状，及时就医')
        
        return {
            'risk_level': risk_level,
            'risk_description': risk_description,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'detection_summary': {
                'afib_detection_rate': f"{afib_rate:.1%}",
                'premature_beats_rate': f"{max(pac_rate, pvc_rate):.1%}",
                'normal_rhythm_rate': f"{detection_stats.get('normal_rhythm_rate', 0):.1%}"
            }
        }
    
    def save_analysis_result(self, device_id, analysis_result):
        """
        保存分析结果到JSON文件
        
        Args:
            device_id: 设备ID
            analysis_result: 分析结果
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{device_id}_vascular_analysis_{timestamp}.json"
        output_path = os.path.join(self.output_dir, output_filename)
        
        try:
            use_slow = os.environ.get('VASCULAR_SLOW_SERIALIZE', '0') == '1'
            if _orjson is not None and not use_slow:
                options = _orjson.OPT_INDENT_2 | _orjson.OPT_SERIALIZE_NUMPY | _orjson.OPT_NON_STR_KEYS
                with open(output_path, 'wb') as f:
                    f.write(_orjson.dumps(analysis_result, option=options))
            else:
                serializable_result = self._make_json_serializable(analysis_result)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_result, f, indent=2, ensure_ascii=False)
            print(f"分析结果已保存: {output_path}")
        except Exception as e:
            print(f"保存分析结果失败 {output_path}: {e}")
    
    def _make_json_serializable(self, obj):
        """
        将对象转换为JSON可序列化的格式
        
        Args:
            obj: 要转换的对象
            
        Returns:
            serializable_obj: JSON可序列化的对象
        """
        if isinstance(obj, dict):
            # 确保字典键为字符串，避免保存时 orjson 报错（Dict key must be str）
            return {str(key): self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bool):
            return obj
        elif obj is None:
            return None
        elif isinstance(obj, (int, float, str)):
            return obj
        else:
            # 对于其他类型，尝试转换为字符串
            return str(obj)
    
    def calculate_segment_statistics(self, segment_results):
        """
        计算数据段统计信息
        
        Args:
            segment_results: 各数据段的分析结果列表
            
        Returns:
            statistics: 统计信息字典
        """
        if not segment_results:
            return {}
        
        # 过滤有效结果
        valid_results = [r for r in segment_results if r is not None and r.get('signal_quality') == 'good']
        
        if not valid_results:
            return {
                'total_segments': len(segment_results),
                'valid_segments': 0,
                'signal_quality_rate': 0.0
            }
        
        # 提取各项指标
        heart_rates = [r['heart_rate'] for r in valid_results if r.get('heart_rate') is not None]
        pwv_values = [r['estimated_pwv'] for r in valid_results if r.get('estimated_pwv') is not None]
        aix_values = [r['augmentation_index'] for r in valid_results if r.get('augmentation_index') is not None]
        vascular_ages = [r['vascular_age'] for r in valid_results if r.get('vascular_age') is not None]
        
        # 形态学特征统计
        rise_time_ratios = []
        systolic_diastolic_ratios = []
        amplitude_variations = []
        
        for r in valid_results:
            morph_features = r.get('morphological_features', {})
            if morph_features.get('rise_time_ratio') is not None:
                rise_time_ratios.append(morph_features['rise_time_ratio'])
            if morph_features.get('systolic_diastolic_ratio') is not None:
                systolic_diastolic_ratios.append(morph_features['systolic_diastolic_ratio'])
            if morph_features.get('amplitude_variation') is not None:
                amplitude_variations.append(morph_features['amplitude_variation'])
        
        # SDPPG特征统计
        aging_indices = []
        for r in valid_results:
            sdppg_features = r.get('sdppg_features', {})
            if sdppg_features.get('aging_index') is not None:
                aging_indices.append(sdppg_features['aging_index'])
        
        # 频域特征统计
        lf_hf_ratios = []
        total_powers = []
        for r in valid_results:
            freq_features = r.get('frequency_domain_features', {})
            if freq_features.get('lf_hf_ratio') is not None:
                lf_hf_ratios.append(freq_features['lf_hf_ratio'])
            if freq_features.get('total_power') is not None:
                total_powers.append(freq_features['total_power'])
        
        # 计算统计值
        def safe_stats(values):
            if not values:
                return {'mean': None, 'std': None, 'min': None, 'max': None, 'count': 0}
            return {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values)
            }
        
        statistics = {
            'total_segments': len(segment_results),
            'valid_segments': len(valid_results),
            'signal_quality_rate': len(valid_results) / len(segment_results),
            
            # 基础生理指标
            'heart_rate_stats': safe_stats(heart_rates),
            'pwv_stats': safe_stats(pwv_values),
            'aix_stats': safe_stats(aix_values),
            'vascular_age_stats': safe_stats(vascular_ages),
            
            # 形态学特征统计
            'rise_time_ratio_stats': safe_stats(rise_time_ratios),
            'systolic_diastolic_ratio_stats': safe_stats(systolic_diastolic_ratios),
            'amplitude_variation_stats': safe_stats(amplitude_variations),
            
            # SDPPG特征统计
            'aging_index_stats': safe_stats(aging_indices),
            
            # 频域特征统计
            'lf_hf_ratio_stats': safe_stats(lf_hf_ratios),
            'total_power_stats': safe_stats(total_powers)
        }
        
        return statistics
    
    def assess_cardiovascular_risk(self, statistics):
        """
        评估心血管风险
        
        Args:
            statistics: 统计信息
            
        Returns:
            risk_assessment: 风险评估结果
        """
        risk_factors = []
        risk_score = 0
        
        # PWV风险评估
        pwv_stats = statistics.get('pwv_stats', {})
        if pwv_stats.get('mean') is not None:
            pwv_mean = pwv_stats['mean']
            if pwv_mean > 12:
                risk_factors.append("PWV显著升高 (>12 m/s)")
                risk_score += 3
            elif pwv_mean > 10:
                risk_factors.append("PWV轻度升高 (>10 m/s)")
                risk_score += 2
            elif pwv_mean > 8:
                risk_factors.append("PWV边界升高 (>8 m/s)")
                risk_score += 1
        
        # AIx风险评估
        aix_stats = statistics.get('aix_stats', {})
        if aix_stats.get('mean') is not None:
            aix_mean = aix_stats['mean']
            if aix_mean > 30:
                risk_factors.append("增强指数显著升高 (>30%)")
                risk_score += 2
            elif aix_mean > 20:
                risk_factors.append("增强指数轻度升高 (>20%)")
                risk_score += 1
        
        # 血管年龄风险评估
        vascular_age_stats = statistics.get('vascular_age_stats', {})
        if vascular_age_stats.get('mean') is not None:
            vascular_age = vascular_age_stats['mean']
            if vascular_age > 70:
                risk_factors.append("血管年龄显著老化 (>70岁)")
                risk_score += 2
            elif vascular_age > 60:
                risk_factors.append("血管年龄轻度老化 (>60岁)")
                risk_score += 1
        
        # 心率变异性风险评估
        hr_stats = statistics.get('heart_rate_stats', {})
        if hr_stats.get('std') is not None:
            hr_variability = hr_stats['std']
            if hr_variability < 2:
                risk_factors.append("心率变异性降低")
                risk_score += 1
        
        # SDPPG老化指数风险评估
        aging_stats = statistics.get('aging_index_stats', {})
        if aging_stats.get('mean') is not None:
            aging_index = aging_stats['mean']
            if aging_index > 0.5:
                risk_factors.append("血管老化指数升高")
                risk_score += 1
        
        # 频域LF/HF比值风险评估
        lf_hf_stats = statistics.get('lf_hf_ratio_stats', {})
        if lf_hf_stats.get('mean') is not None:
            lf_hf_ratio = lf_hf_stats['mean']
            if lf_hf_ratio > 4:
                risk_factors.append("自主神经平衡失调")
                risk_score += 1
        
        # 风险等级评估
        if risk_score >= 6:
            risk_level = "高风险"
        elif risk_score >= 3:
            risk_level = "中等风险"
        elif risk_score >= 1:
            risk_level = "低风险"
        else:
            risk_level = "正常"
        
        risk_assessment = {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendations': self.generate_recommendations(risk_level, risk_factors)
        }
        
        return risk_assessment
    
    def generate_recommendations(self, risk_level, risk_factors):
        """
        生成健康建议
        
        Args:
            risk_level: 风险等级
            risk_factors: 风险因素列表
            
        Returns:
            recommendations: 建议列表
        """
        recommendations = []
        
        if risk_level == "高风险":
            recommendations.extend([
                "建议尽快就医，进行详细的心血管检查",
                "严格控制血压、血糖、血脂",
                "戒烟限酒，改善生活方式",
                "在医生指导下进行适量运动"
            ])
        elif risk_level == "中等风险":
            recommendations.extend([
                "建议定期监测血管功能",
                "保持健康的生活方式",
                "适量有氧运动，如快走、游泳",
                "控制体重，均衡饮食"
            ])
        elif risk_level == "低风险":
            recommendations.extend([
                "继续保持良好的生活习惯",
                "定期进行健康体检",
                "适量运动，保持心血管健康"
            ])
        else:
            recommendations.extend([
                "血管功能良好，继续保持",
                "定期监测，预防心血管疾病"
            ])
        
        # 针对特定风险因素的建议
        for factor in risk_factors:
            if "PWV" in factor:
                recommendations.append("重点关注动脉硬化预防")
            if "增强指数" in factor:
                recommendations.append("注意大动脉弹性维护")
            if "血管年龄" in factor:
                recommendations.append("采取抗衰老措施")
            if "心率变异性" in factor:
                recommendations.append("改善自主神经功能")
        
        return list(set(recommendations))  # 去重
    
    def analyze_inflammation_segment(self, decompressed_data, segment_index, segment_info):
        """
        分析单个数据段的炎症指标
        
        Args:
            decompressed_data: 解压缩的PPG数据
            segment_index: 数据段索引
            segment_info: 数据段信息
            
        Returns:
            inflammation_result: 炎症分析结果
        """
        try:
            ppg_signal = decompressed_data if isinstance(decompressed_data, np.ndarray) and decompressed_data.dtype == np.float64 else np.array(decompressed_data, dtype=np.float64)
            
            # 使用炎症检测器进行综合分析
            inflammation_analysis = self.inflammation_detector.comprehensive_inflammation_analysis(ppg_signal)
            
            if 'error' in inflammation_analysis:
                return None
            
            # 组装结果
            result = {
                'segment_index': segment_index,
                'collect_time': segment_info.get('collectTime'),
                'create_time': segment_info.get('createTime'),
                'signal_length': len(ppg_signal),
                'perfusion_analysis': inflammation_analysis.get('perfusion_analysis', {}),
                'morphology_analysis': inflammation_analysis.get('morphology_analysis', {}),
                'hrv_analysis': inflammation_analysis.get('hrv_analysis', {}),
                'inflammation_assessment': inflammation_analysis.get('inflammation_assessment', {}),
                'predicted_markers': inflammation_analysis.get('predicted_markers', {}),
                'clinical_recommendations': inflammation_analysis.get('clinical_recommendations', {}),
                'analysis_quality': inflammation_analysis.get('analysis_quality', 'poor')
            }
            
            return result
            
        except Exception as e:
            print(f"炎症分析段 {segment_index} 时出错: {e}")
            return None
    
    def calculate_inflammation_statistics(self, inflammation_results):
        """
        计算炎症分析统计信息
        
        Args:
            inflammation_results: 炎症分析结果列表
            
        Returns:
            statistics: 统计信息
        """
        if not inflammation_results:
            return {
                'total_segments': 0,
                'average_inflammation_score': 0,
                'inflammation_grade_distribution': {},
                'perfusion_statistics': {},
                'hrv_statistics': {},
                'predicted_markers_statistics': {},
                'analysis_quality_distribution': {}
            }
        
        try:
            # 提取各项指标
            inflammation_scores = []
            inflammation_grades = []
            perfusion_indices = []
            perfusion_variabilities = []
            sdnn_values = []
            rmssd_values = []
            lf_hf_ratios = []
            predicted_crp = []
            predicted_il6 = []
            analysis_qualities = []
            
            for result in inflammation_results:
                # 炎症评分和等级
                assessment = result.get('inflammation_assessment', {})
                if 'inflammation_score' in assessment:
                    inflammation_scores.append(assessment['inflammation_score'])
                if 'inflammation_grade' in assessment:
                    inflammation_grades.append(assessment['inflammation_grade'])
                
                # 灌注指标
                perfusion = result.get('perfusion_analysis', {})
                if 'perfusion_index' in perfusion:
                    perfusion_indices.append(perfusion['perfusion_index'])
                if 'perfusion_variability' in perfusion:
                    perfusion_variabilities.append(perfusion['perfusion_variability'])
                
                # HRV指标
                hrv = result.get('hrv_analysis', {})
                if 'sdnn' in hrv:
                    sdnn_values.append(hrv['sdnn'])
                if 'rmssd' in hrv:
                    rmssd_values.append(hrv['rmssd'])
                if 'lf_hf_ratio' in hrv:
                    lf_hf_ratios.append(hrv['lf_hf_ratio'])
                
                # 预测标志物
                markers = result.get('predicted_markers', {})
                if 'predicted_crp' in markers:
                    predicted_crp.append(markers['predicted_crp'])
                if 'predicted_il6' in markers:
                    predicted_il6.append(markers['predicted_il6'])
                
                # 分析质量
                quality = result.get('analysis_quality', 'poor')
                analysis_qualities.append(quality)
            
            # 计算统计信息
            statistics = {
                'total_segments': len(inflammation_results),
                'average_inflammation_score': np.mean(inflammation_scores) if inflammation_scores else 0,
                'inflammation_score_std': np.std(inflammation_scores) if inflammation_scores else 0,
                'inflammation_grade_distribution': self._calculate_distribution(inflammation_grades),
                'perfusion_statistics': {
                    'average_perfusion_index': np.mean(perfusion_indices) if perfusion_indices else 0,
                    'perfusion_index_std': np.std(perfusion_indices) if perfusion_indices else 0,
                    'average_perfusion_variability': np.mean(perfusion_variabilities) if perfusion_variabilities else 0,
                    'perfusion_variability_std': np.std(perfusion_variabilities) if perfusion_variabilities else 0
                },
                'hrv_statistics': {
                    'average_sdnn': np.mean(sdnn_values) if sdnn_values else 0,
                    'sdnn_std': np.std(sdnn_values) if sdnn_values else 0,
                    'average_rmssd': np.mean(rmssd_values) if rmssd_values else 0,
                    'rmssd_std': np.std(rmssd_values) if rmssd_values else 0,
                    'average_lf_hf_ratio': np.mean(lf_hf_ratios) if lf_hf_ratios else 0,
                    'lf_hf_ratio_std': np.std(lf_hf_ratios) if lf_hf_ratios else 0
                },
                'predicted_markers_statistics': {
                    'average_predicted_crp': np.mean(predicted_crp) if predicted_crp else 0,
                    'predicted_crp_std': np.std(predicted_crp) if predicted_crp else 0,
                    'average_predicted_il6': np.mean(predicted_il6) if predicted_il6 else 0,
                    'predicted_il6_std': np.std(predicted_il6) if predicted_il6 else 0
                },
                'analysis_quality_distribution': self._calculate_distribution(analysis_qualities)
            }
            
            return statistics
            
        except Exception as e:
            print(f"计算炎症统计信息时出错: {e}")
            return {
                'total_segments': len(inflammation_results),
                'error': str(e)
            }
    
    def _calculate_distribution(self, values):
        """计算值的分布"""
        if not values:
            return {}
        
        from collections import Counter
        distribution = Counter(values)
        total = len(values)
        
        return {key: {'count': count, 'percentage': count / total * 100} 
                for key, count in distribution.items()}
    
    def assess_inflammation_risk(self, inflammation_statistics):
        """
        评估炎症风险
        
        Args:
            inflammation_statistics: 炎症统计信息
            
        Returns:
            risk_assessment: 风险评估结果
        """
        try:
            if inflammation_statistics.get('total_segments', 0) == 0:
                return {
                    'overall_risk_level': 'unknown',
                    'risk_score': 0,
                    'risk_factors': [],
                    'recommendations': ['数据不足，无法评估炎症风险']
                }
            
            avg_score = inflammation_statistics.get('average_inflammation_score', 0)
            grade_dist = inflammation_statistics.get('inflammation_grade_distribution', {})
            perfusion_stats = inflammation_statistics.get('perfusion_statistics', {})
            hrv_stats = inflammation_statistics.get('hrv_statistics', {})
            
            risk_factors = []
            risk_score = 0
            
            # 基于平均炎症评分评估风险
            if avg_score >= 6:
                risk_score += 40
                risk_factors.append("高炎症评分")
            elif avg_score >= 4:
                risk_score += 25
                risk_factors.append("中等炎症评分")
            elif avg_score >= 2:
                risk_score += 10
                risk_factors.append("轻度炎症评分")
            
            # 基于炎症等级分布评估
            severe_percentage = grade_dist.get('severe', {}).get('percentage', 0)
            moderate_percentage = grade_dist.get('moderate', {}).get('percentage', 0)
            
            if severe_percentage > 20:
                risk_score += 30
                risk_factors.append("重度炎症比例高")
            elif moderate_percentage > 40:
                risk_score += 20
                risk_factors.append("中度炎症比例高")
            
            # 基于灌注指标评估
            avg_perfusion = perfusion_stats.get('average_perfusion_index', 0)
            avg_variability = perfusion_stats.get('average_perfusion_variability', 0)
            
            if avg_perfusion < 0.2:
                risk_score += 15
                risk_factors.append("灌注指数低")
            
            if avg_variability > 25:
                risk_score += 15
                risk_factors.append("灌注变异性高")
            
            # 基于HRV指标评估
            avg_sdnn = hrv_stats.get('average_sdnn', 0)
            avg_rmssd = hrv_stats.get('average_rmssd', 0)
            avg_lf_hf = hrv_stats.get('average_lf_hf_ratio', 0)
            
            if avg_sdnn < 30:
                risk_score += 10
                risk_factors.append("SDNN低")
            
            if avg_rmssd < 20:
                risk_score += 10
                risk_factors.append("RMSSD低")
            
            if avg_lf_hf > 5:
                risk_score += 10
                risk_factors.append("LF/HF比值高")
            
            # 确定风险等级
            if risk_score >= 70:
                risk_level = 'critical'
            elif risk_score >= 50:
                risk_level = 'high'
            elif risk_score >= 30:
                risk_level = 'moderate'
            elif risk_score >= 15:
                risk_level = 'mild'
            else:
                risk_level = 'low'
            
            # 生成建议
            recommendations = self.generate_inflammation_recommendations(risk_level, risk_factors)
            
            return {
                'overall_risk_level': risk_level,
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'recommendations': recommendations,
                'detailed_assessment': {
                    'inflammation_score_risk': avg_score,
                    'perfusion_risk': avg_perfusion < 0.2 or avg_variability > 25,
                    'hrv_risk': avg_sdnn < 30 or avg_rmssd < 20 or avg_lf_hf > 5,
                    'grade_distribution_risk': severe_percentage > 20 or moderate_percentage > 40
                }
            }
            
        except Exception as e:
            return {
                'overall_risk_level': 'unknown',
                'risk_score': 0,
                'risk_factors': [],
                'recommendations': [f'风险评估出错: {str(e)}']
            }
    
    def generate_inflammation_recommendations(self, risk_level, risk_factors):
        """
        生成炎症相关建议
        
        Args:
            risk_level: 风险等级
            risk_factors: 风险因素列表
            
        Returns:
            recommendations: 建议列表
        """
        recommendations = []
        
        # 基于风险等级的基础建议
        if risk_level == 'critical':
            recommendations.extend([
                "立即进行全面炎症评估",
                "考虑抗炎治疗",
                "密切监测炎症标志物",
                "避免剧烈活动"
            ])
        elif risk_level == 'high':
            recommendations.extend([
                "建议血液检查确认炎症标志物",
                "增加监测频率",
                "注意休息和营养"
            ])
        elif risk_level == 'moderate':
            recommendations.extend([
                "定期监测炎症指标",
                "保持健康生活方式",
                "适当抗氧化剂补充"
            ])
        elif risk_level == 'mild':
            recommendations.extend([
                "注意预防感染",
                "保持充足睡眠",
                "适度运动"
            ])
        else:
            recommendations.append("继续保持健康状态")
        
        # 基于具体风险因素的建议
        for factor in risk_factors:
            if "炎症评分" in factor:
                recommendations.append("重点关注炎症控制")
            if "灌注" in factor:
                recommendations.append("改善微循环功能")
            if "HRV" in factor or "SDNN" in factor or "RMSSD" in factor:
                recommendations.append("注意自主神经功能调节")
            if "LF/HF" in factor:
                recommendations.append("减少交感神经激活")
        
        return list(set(recommendations))  # 去重
    
    def process_device_data(self, file_path):
        """
        处理单个设备的数据文件，包含血管功能分析和血流分析
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            analysis_result: 分析结果
        """
        if os.environ.get('VASCULAR_VERBOSE', '0') == '1':
            print(f"正在处理文件: {file_path}")
        
        # 加载JSON数据
        data = self.load_json_file(file_path)
        if data is None:
            return None
        
        device_id = data.get('deviceId')
        if not device_id:
            print(f"文件中未找到deviceId: {file_path}")
            return None
        
        processed_data = data.get('processedData', [])
        if not processed_data:
            print(f"文件中未找到processedData: {file_path}")
            return None
        
        if os.environ.get('VASCULAR_VERBOSE', '0') == '1':
            print(f"设备ID: {device_id}, 数据段数量: {len(processed_data)}")
        
        # 分析每个数据段
        segment_results = []
        blood_flow_results = []  # 血流分析结果
        arrhythmia_results = []  # 心律不齐分析结果
        inflammation_results = []  # 炎症分析结果
        
        # 收集所有数据用于睡眠分析
        all_ppg_data = []
        all_timestamps = []
        
        analyzer = self.analyzer
        blood_estimator = self.blood_flow_estimator
        arr_detector = self.arrhythmia_detector
        inflam_detector = self.inflammation_detector
        if os.environ.get('SEGMENT_PARALLEL', '1') == '1':
            seg_exec = os.environ.get('SEGMENT_EXECUTOR', 'thread')
            default_workers = max(8, min(32, (os.cpu_count() or 2) * 2))
            env_workers = os.environ.get('SEGMENT_PARALLEL_WORKERS')
            max_workers = default_workers if not (env_workers and env_workers.isdigit()) else max(1, int(env_workers))
            def _work(i, segment):
                decompressed_data = segment.get('decompressedData', [])
                if not decompressed_data or len(decompressed_data) < 100:
                    return None
                ppg_signal = np.asarray(decompressed_data, dtype=np.float64)
                vr = analyzer.analyze_ppg_segment(ppg_signal)
                if vr is not None:
                    vr['segment_index'] = i
                    vr['collect_time'] = segment.get('collectTime')
                    vr['create_time'] = segment.get('createTime')
                bfr = self.analyze_blood_flow_segment(ppg_signal, i, segment)
                arr = self.analyze_arrhythmia_segment(ppg_signal, i, segment)
                inflam = self.analyze_inflammation_segment(ppg_signal, i, segment)
                ct = segment.get('collectTime')
                ts_out = None
                if os.environ.get('SKIP_SLEEP_ANALYSIS', '0') != '1' and ct:
                    base_time = datetime.strptime(ct, "%Y-%m-%d %H:%M:%S")
                    n = len(decompressed_data)
                    base_np = np.datetime64(base_time, 'ms')
                    offsets = self._get_offsets(n)
                    times = base_np + offsets
                    if os.environ.get('TS_STRING_EAGER', '0') == '1':
                        ts_out = np.char.replace(np.datetime_as_string(times, unit='s'), 'T', ' ').tolist()
                    else:
                        ts_out = times
                return (vr, bfr, arr, inflam, ppg_signal, ts_out)
            if seg_exec == 'process':
                from concurrent.futures import ProcessPoolExecutor, as_completed
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(_work, i, seg) for i, seg in enumerate(processed_data)]
                    done = 0
                    for fut in as_completed(futures):
                        try:
                            out = fut.result()
                            if out is None:
                                continue
                            vr, bfr, arr, inflam, data_chunk, ts = out
                            if vr is not None:
                                segment_results.append(vr)
                            if bfr is not None:
                                blood_flow_results.append(bfr)
                            if arr is not None:
                                arrhythmia_results.append(arr)
                            if inflam is not None:
                                inflammation_results.append(inflam)
                            all_ppg_data.append(data_chunk)
                            if ts is not None:
                                if isinstance(ts, np.ndarray):
                                    all_timestamps.append(ts)
                                elif isinstance(ts, list):
                                    all_timestamps.extend(ts)
                            done += 1
                            if os.environ.get('VASCULAR_VERBOSE', '0') == '1' and (done % 10 == 0):
                                print(f"已处理 {done}/{len(processed_data)} 个数据段")
                        except Exception as e:
                            print(f"并行处理数据段时出错: {e}")
                            continue
            else:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(_work, i, seg) for i, seg in enumerate(processed_data)]
                    done = 0
                    for fut in as_completed(futures):
                        try:
                            out = fut.result()
                            if out is None:
                                continue
                            vr, bfr, arr, inflam, data_chunk, ts = out
                            if vr is not None:
                                segment_results.append(vr)
                            if bfr is not None:
                                blood_flow_results.append(bfr)
                            if arr is not None:
                                arrhythmia_results.append(arr)
                            if inflam is not None:
                                inflammation_results.append(inflam)
                            all_ppg_data.append(data_chunk)
                            if ts is not None:
                                if isinstance(ts, np.ndarray):
                                    all_timestamps.append(ts)
                                elif isinstance(ts, list):
                                    all_timestamps.extend(ts)
                            done += 1
                            if os.environ.get('VASCULAR_VERBOSE', '0') == '1' and (done % 10 == 0):
                                print(f"已处理 {done}/{len(processed_data)} 个数据段")
                        except Exception as e:
                            print(f"并行处理数据段时出错: {e}")
                            continue
        else:
            for i, segment in enumerate(processed_data):
                try:
                    decompressed_data = segment.get('decompressedData', [])
                    if not decompressed_data or len(decompressed_data) < 100:
                        continue
                    ppg_signal = np.asarray(decompressed_data, dtype=np.float64)
                    vascular_result = analyzer.analyze_ppg_segment(ppg_signal)
                    if vascular_result is not None:
                        vascular_result['segment_index'] = i
                        vascular_result['collect_time'] = segment.get('collectTime')
                        vascular_result['create_time'] = segment.get('createTime')
                        segment_results.append(vascular_result)
                    blood_flow_result = self.analyze_blood_flow_segment(ppg_signal, i, segment)
                    if blood_flow_result is not None:
                        blood_flow_results.append(blood_flow_result)
                    arrhythmia_result = self.analyze_arrhythmia_segment(ppg_signal, i, segment)
                    if arrhythmia_result is not None:
                        arrhythmia_results.append(arrhythmia_result)
                    inflammation_result = self.analyze_inflammation_segment(ppg_signal, i, segment)
                    if inflammation_result is not None:
                        inflammation_results.append(inflammation_result)
                    all_ppg_data.append(ppg_signal)
                    collect_time = segment.get('collectTime')
                    if os.environ.get('SKIP_SLEEP_ANALYSIS', '0') != '1' and collect_time:
                        base_time = datetime.strptime(collect_time, "%Y-%m-%d %H:%M:%S")
                        n = len(decompressed_data)
                        base_np = np.datetime64(base_time, 'ms')
                        offsets = self._get_offsets(n)
                        times = base_np + offsets
                        if os.environ.get('TS_STRING_EAGER', '0') == '1':
                            strings = np.char.replace(np.datetime_as_string(times, unit='s'), 'T', ' ').tolist()
                            all_timestamps.extend(strings)
                        else:
                            all_timestamps.append(times)
                    if os.environ.get('VASCULAR_VERBOSE', '0') == '1' and ((i + 1) % 10 == 0):
                        print(f"已处理 {i + 1}/{len(processed_data)} 个数据段")
                except Exception as e:
                    print(f"处理数据段 {i} 时出错: {e}")
                    continue
        
        if os.environ.get('VASCULAR_VERBOSE', '0') == '1':
            print(f"成功分析 {len(segment_results)} 个血管功能数据段")
            print(f"成功分析 {len(blood_flow_results)} 个血流数据段")
            print(f"成功分析 {len(arrhythmia_results)} 个心律不齐数据段")
            print(f"成功分析 {len(inflammation_results)} 个炎症数据段")
        
        # 计算血管功能统计信息
        vascular_statistics = self.calculate_segment_statistics(segment_results)
        
        # 计算血流统计信息
        blood_flow_statistics = self.calculate_blood_flow_statistics(blood_flow_results)
        
        # 计算心律不齐统计信息
        arrhythmia_statistics = self.calculate_arrhythmia_statistics(arrhythmia_results)
        
        # 计算炎症统计信息
        inflammation_statistics = self.calculate_inflammation_statistics(inflammation_results)
        
        
        # 睡眠分析
        sleep_analysis_result = None
        if os.environ.get('SKIP_SLEEP_ANALYSIS', '0') == '1':
            sleep_analysis_result = self._get_default_sleep_result()
        elif all_ppg_data and all_timestamps:
            try:
                max_points = int(os.environ.get('SLEEP_MAX_POINTS', '200000') or '200000')
                sample_rate = float(os.environ.get('SLEEP_SAMPLE_RATE', '0') or '0')
                xs = np.concatenate(all_ppg_data) if all_ppg_data and isinstance(all_ppg_data[0], np.ndarray) else np.array(all_ppg_data)
                ts_arr = np.concatenate(all_timestamps) if isinstance(all_timestamps, list) and all_timestamps and isinstance(all_timestamps[0], np.ndarray) else None
                n = xs.shape[0]
                if max_points > 0 and n > max_points:
                    step = max(1, n // max_points)
                    xs = xs[::step]
                    ts_arr = ts_arr[::step]
                elif 0 < sample_rate < 1.0:
                    step = max(1, int(1.0 / sample_rate))
                    xs = xs[::step]
                    if ts_arr is not None:
                        ts_arr = ts_arr[::step]
                sleep_analysis_result = analyze_sleep_data(xs, ts_arr if ts_arr is not None else all_timestamps, sampling_rate=125)
            except Exception:
                sleep_analysis_result = self._get_default_sleep_result()
        else:
            sleep_analysis_result = self._get_default_sleep_result()
        
        # 评估心血管风险
        risk_assessment = self.assess_cardiovascular_risk(vascular_statistics)
        
        # 血流风险评估
        blood_flow_risk = self.assess_blood_flow_risk(blood_flow_statistics)
        
        # 心律不齐风险评估
        arrhythmia_risk = self.assess_arrhythmia_risk(arrhythmia_statistics)
        
        # 炎症风险评估
        inflammation_risk = self.assess_inflammation_risk(inflammation_statistics)
        
        # 组装最终结果
        analysis_result = {
            'device_id': device_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_segments': len(processed_data),
                'analyzed_vascular_segments': len(segment_results),
                'analyzed_blood_flow_segments': len(blood_flow_results),
                'analyzed_arrhythmia_segments': len(arrhythmia_results),
                'analyzed_inflammation_segments': len(inflammation_results),
                'sleep_analysis_data_points': len(all_ppg_data),
                'vascular_analysis_success_rate': len(segment_results) / len(processed_data) if processed_data else 0,
                'blood_flow_analysis_success_rate': len(blood_flow_results) / len(processed_data) if processed_data else 0,
                'arrhythmia_analysis_success_rate': len(arrhythmia_results) / len(processed_data) if processed_data else 0,
                'inflammation_analysis_success_rate': len(inflammation_results) / len(processed_data) if processed_data else 0,
        
            },
            'vascular_function_statistics': vascular_statistics,
            'blood_flow_statistics': blood_flow_statistics,
            'arrhythmia_statistics': arrhythmia_statistics,
            'inflammation_statistics': inflammation_statistics,
            
            'sleep_analysis': sleep_analysis_result,
            'cardiovascular_risk_assessment': risk_assessment,
            'blood_flow_risk_assessment': blood_flow_risk,
            'arrhythmia_risk_assessment': arrhythmia_risk,
            'inflammation_risk_assessment': inflammation_risk,
            
            'detailed_vascular_analysis': segment_results[:50],  # 只保存前50个详细结果
            'detailed_blood_flow_analysis': blood_flow_results[:50],  # 只保存前50个详细结果
            'detailed_arrhythmia_analysis': arrhythmia_results[:50],  # 只保存前50个详细结果
            'detailed_inflammation_analysis': inflammation_results[:50],  # 只保存前50个详细结果
            
        }
        
        return analysis_result

    def process_device_payload(self, data: Dict[str, Any]):
        """直接处理内存中的设备数据payload，避免中间JSON落盘开销"""
        try:
            device_id = data.get('deviceId') or data.get('device_id')
            if not device_id:
                return None
            processed_data = data.get('processedData') or data.get('processed_data') or []
            if not processed_data:
                return None
            segment_results = []
            blood_flow_results = []
            arrhythmia_results = []
            inflammation_results = []
            all_ppg_data = []
            all_timestamps = []
            analyzer = self.analyzer
            if os.environ.get('SEGMENT_PARALLEL', '1') == '1':
                seg_exec = os.environ.get('SEGMENT_EXECUTOR', 'thread')
                default_workers = 8
                env_workers = os.environ.get('SEGMENT_PARALLEL_WORKERS')
                max_workers = default_workers if not (env_workers and env_workers.isdigit()) else max(1, int(env_workers))
                def _work(i, segment):
                    decompressed_data = segment.get('decompressedData', [])
                    if not decompressed_data or len(decompressed_data) < 100:
                        return None
                    ppg_signal = np.asarray(decompressed_data, dtype=np.float64)
                    vr = analyzer.analyze_ppg_segment(ppg_signal)
                    if vr is not None:
                        vr['segment_index'] = i
                        vr['collect_time'] = segment.get('collectTime')
                        vr['create_time'] = segment.get('createTime')
                    bfr = self.analyze_blood_flow_segment(ppg_signal, i, segment)
                    arr = self.analyze_arrhythmia_segment(ppg_signal, i, segment)
                    inflam = self.analyze_inflammation_segment(ppg_signal, i, segment)
                    ct = segment.get('collectTime')
                    times = None
                    if ct:
                        base_time = datetime.strptime(ct, "%Y-%m-%d %H:%M:%S")
                        n = len(decompressed_data)
                        base_np = np.datetime64(base_time, 'ms')
                        offsets = np.arange(n, dtype='timedelta64[ms]') * 8
                        times = base_np + offsets
                    return (vr, bfr, arr, inflam, decompressed_data, times)
                if seg_exec == 'process':
                    from concurrent.futures import ProcessPoolExecutor, as_completed
                    with ProcessPoolExecutor(max_workers=max_workers) as ex:
                        futs = [ex.submit(_work, i, seg) for i, seg in enumerate(processed_data)]
                        for fut in as_completed(futs):
                            try:
                                out = fut.result()
                                if out is None:
                                    continue
                                vr, bfr, arr, inflam, data_chunk, ts = out
                                if vr is not None:
                                    segment_results.append(vr)
                                if bfr is not None:
                                    blood_flow_results.append(bfr)
                                if arr is not None:
                                    arrhythmia_results.append(arr)
                                if inflam is not None:
                                    inflammation_results.append(inflam)
                                all_ppg_data.append(data_chunk)
                                if ts is not None:
                                    if isinstance(ts, np.ndarray):
                                        all_timestamps.append(ts)
                                    elif isinstance(ts, list):
                                        all_timestamps.extend(ts)
                            except Exception:
                                continue
                else:
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    with ThreadPoolExecutor(max_workers=max_workers) as ex:
                        futs = [ex.submit(_work, i, seg) for i, seg in enumerate(processed_data)]
                        for fut in as_completed(futs):
                            try:
                                out = fut.result()
                                if out is None:
                                    continue
                                vr, bfr, arr, inflam, data_chunk, ts = out
                                if vr is not None:
                                    segment_results.append(vr)
                                if bfr is not None:
                                    blood_flow_results.append(bfr)
                                if arr is not None:
                                    arrhythmia_results.append(arr)
                                if inflam is not None:
                                    inflammation_results.append(inflam)
                                all_ppg_data.append(data_chunk)
                                if ts is not None:
                                    if isinstance(ts, np.ndarray):
                                        all_timestamps.append(ts)
                                    elif isinstance(ts, list):
                                        all_timestamps.extend(ts)
                            except Exception:
                                continue
            else:
                for i, segment in enumerate(processed_data):
                    try:
                        decompressed_data = segment.get('decompressedData', [])
                        if not decompressed_data or len(decompressed_data) < 100:
                            continue
                        ppg_signal = np.asarray(decompressed_data, dtype=np.float64)
                        vascular_result = analyzer.analyze_ppg_segment(ppg_signal)
                        if vascular_result is not None:
                            vascular_result['segment_index'] = i
                            vascular_result['collect_time'] = segment.get('collectTime')
                            vascular_result['create_time'] = segment.get('createTime')
                            segment_results.append(vascular_result)
                        blood_flow_result = self.analyze_blood_flow_segment(ppg_signal, i, segment)
                        if blood_flow_result is not None:
                            blood_flow_results.append(blood_flow_result)
                        arrhythmia_result = self.analyze_arrhythmia_segment(ppg_signal, i, segment)
                        if arrhythmia_result is not None:
                            arrhythmia_results.append(arrhythmia_result)
                        inflammation_result = self.analyze_inflammation_segment(ppg_signal, i, segment)
                        if inflammation_result is not None:
                            inflammation_results.append(inflammation_result)
                        all_ppg_data.append(ppg_signal)
                        collect_time = segment.get('collectTime')
                        if os.environ.get('SKIP_SLEEP_ANALYSIS', '0') != '1' and collect_time:
                            base_time = datetime.strptime(collect_time, "%Y-%m-%d %H:%M:%S")
                            n = len(decompressed_data)
                            base_np = np.datetime64(base_time, 'ms')
                            offsets = self._get_offsets(n)
                            times = base_np + offsets
                            if os.environ.get('TS_STRING_EAGER', '0') == '1':
                                strings = np.char.replace(np.datetime_as_string(times, unit='s'), 'T', ' ').tolist()
                                all_timestamps.extend(strings)
                            else:
                                all_timestamps.append(times)
                    except Exception:
                        continue
            sleep_analysis_result = None
            if os.environ.get('SKIP_SLEEP_ANALYSIS', '0') == '1':
                sleep_analysis_result = self._get_default_sleep_result()
            elif all_ppg_data and all_timestamps:
                try:
                    xs = np.concatenate(all_ppg_data) if all_ppg_data and isinstance(all_ppg_data[0], np.ndarray) else np.array(all_ppg_data)
                    if isinstance(all_timestamps[0], np.ndarray):
                        ts_arr = np.concatenate(all_timestamps)
                        sleep_analysis_result = analyze_sleep_data(xs, ts_arr, sampling_rate=125)
                    else:
                        sleep_analysis_result = analyze_sleep_data(xs, all_timestamps, sampling_rate=125)
                except Exception:
                    sleep_analysis_result = self._get_default_sleep_result()
            else:
                sleep_analysis_result = self._get_default_sleep_result()
            vascular_statistics = self.calculate_segment_statistics(segment_results)
            blood_flow_statistics = self.calculate_blood_flow_statistics(blood_flow_results)
            arrhythmia_statistics = self.calculate_arrhythmia_statistics(arrhythmia_results)
            inflammation_statistics = self.calculate_inflammation_statistics(inflammation_results)
            risk_assessment = self.assess_cardiovascular_risk(vascular_statistics)
            blood_flow_risk = self.assess_blood_flow_risk(blood_flow_statistics)
            arrhythmia_risk = self.assess_arrhythmia_risk(arrhythmia_statistics)
            inflammation_risk = self.assess_inflammation_risk(inflammation_statistics)
            return {
                'device_id': device_id,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_summary': {
                    'total_segments': len(processed_data),
                    'analyzed_vascular_segments': len(segment_results),
                    'analyzed_blood_flow_segments': len(blood_flow_results),
                    'analyzed_arrhythmia_segments': len(arrhythmia_results),
                    'analyzed_inflammation_segments': len(inflammation_results),
                    'sleep_analysis_data_points': len(all_ppg_data),
                },
                'vascular_function_statistics': vascular_statistics,
                'blood_flow_statistics': blood_flow_statistics,
                'arrhythmia_statistics': arrhythmia_statistics,
                'inflammation_statistics': inflammation_statistics,
                'sleep_analysis': sleep_analysis_result,
                'cardiovascular_risk_assessment': risk_assessment,
                'blood_flow_risk_assessment': blood_flow_risk,
                'arrhythmia_risk_assessment': arrhythmia_risk,
                'inflammation_risk_assessment': inflammation_risk,
                'detailed_vascular_analysis': segment_results[:50],
                'detailed_blood_flow_analysis': blood_flow_results[:50],
                'detailed_arrhythmia_analysis': arrhythmia_results[:50],
                'detailed_inflammation_analysis': inflammation_results[:50],
            }
        except Exception:
            return None
    
    def analyze_arrhythmia_segment(self, decompressed_data, segment_index, segment_info):
        """
        分析单个数据段的心律不齐参数
        
        Args:
            decompressed_data: 解压缩的PPG数据
            segment_index: 数据段索引
            segment_info: 数据段信息
            
        Returns:
            arrhythmia_result: 心律不齐分析结果
        """
        try:
            ppg_signal = decompressed_data if isinstance(decompressed_data, np.ndarray) else np.asarray(decompressed_data, dtype=np.float64)
            
            # 综合心律不齐分析
            arrhythmia_analysis = self.arrhythmia_detector.comprehensive_arrhythmia_analysis(ppg_signal)
            
            # 组装结果 - 正确提取各个部分
            result = {
                'segment_index': segment_index,
                'collect_time': segment_info.get('collectTime'),
                'create_time': segment_info.get('createTime'),
                'signal_quality': arrhythmia_analysis.get('signal_quality', 0.0),
                'ppi_analysis': {
                    'peak_count': arrhythmia_analysis.get('peak_count', 0),
                    'ppi_count': arrhythmia_analysis.get('ppi_count', 0),
                    'analysis_status': arrhythmia_analysis.get('analysis_status', '未知')
                },
                'time_domain_features': arrhythmia_analysis.get('time_domain_features', {}),
                'frequency_domain_features': arrhythmia_analysis.get('frequency_domain_features', {}),
                'arrhythmia_detection': {
                    'arrhythmia_detected': arrhythmia_analysis.get('arrhythmia_detected', False),
                    'atrial_fibrillation': arrhythmia_analysis.get('atrial_fibrillation', {}),
                    'premature_beats': arrhythmia_analysis.get('premature_beats', {}),
                    'sinus_arrhythmia': arrhythmia_analysis.get('sinus_arrhythmia', {})
                },
                'risk_assessment': arrhythmia_analysis.get('risk_assessment', {})
            }
            
            return result
            
        except Exception as e:
            print(f"心律不齐分析失败 (段 {segment_index}): {e}")
            return None
    
    def analyze_blood_flow_segment(self, decompressed_data, segment_index, segment_info):
        """
        分析单个数据段的血流参数
        
        Args:
            decompressed_data: 解压缩的PPG数据
            segment_index: 数据段索引
            segment_info: 数据段信息
            
        Returns:
            blood_flow_result: 血流分析结果
        """
        try:
            ppg_signal = decompressed_data if isinstance(decompressed_data, np.ndarray) else np.asarray(decompressed_data, dtype=np.float64)
            
            # 综合血流分析
            flow_analysis = self.blood_flow_estimator.comprehensive_flow_analysis(ppg_signal)
            
            # 计算灌注指数
            pi, flow_status = self.blood_flow_estimator.calculate_perfusion_index(ppg_signal)
            
            # Windkessel模型分析
            windkessel_result = self.blood_flow_estimator.windkessel_model(ppg_signal)
            
            # 提取PPG特征
            ppg_features = self.blood_flow_estimator.extract_ppg_features(ppg_signal)
            
            # 估算绝对血流速度
            estimated_velocity = self.blood_flow_estimator.estimate_absolute_flow_velocity(ppg_signal)
            
            # 组装结果
            blood_flow_result = {
                'segment_index': segment_index,
                'collect_time': segment_info.get('collectTime'),
                'create_time': segment_info.get('createTime'),
                'data_length': len(ppg_signal),
                'perfusion_analysis': {
                    'perfusion_index': pi,
                    'flow_status': flow_status,
                    'flow_classification': self.blood_flow_estimator.classify_flow_status(pi)
                },
                'hemodynamic_modeling': windkessel_result,
                'estimated_flow_velocity': estimated_velocity,
                'ppg_features': ppg_features,
                'comprehensive_analysis': flow_analysis
            }
            
            return blood_flow_result
            
        except Exception as e:
            print(f"血流分析段 {segment_index} 时出错: {e}")
            return None
    
    def calculate_blood_flow_statistics(self, blood_flow_results):
        """
        计算血流分析的统计信息
        
        Args:
            blood_flow_results: 血流分析结果列表
            
        Returns:
            statistics: 统计信息字典
        """
        if not blood_flow_results:
            return {}
        
        # 提取各项指标
        perfusion_indices = []
        flow_velocities = []
        flow_velocity_indices = []
        cardiac_output_indices = []
        vascular_compliances = []
        vascular_resistances = []
        
        flow_status_counts = {'高灌注': 0, '正常灌注': 0, '低灌注': 0}
        
        for result in blood_flow_results:
            # 灌注指数
            perfusion_analysis = result.get('perfusion_analysis', {})
            pi = perfusion_analysis.get('perfusion_index')
            if pi is not None:
                perfusion_indices.append(pi)
            
            flow_status = perfusion_analysis.get('flow_status', '无法评估')
            if flow_status in flow_status_counts:
                flow_status_counts[flow_status] += 1
            
            # 估算血流速度
            velocity = result.get('estimated_flow_velocity')
            if velocity is not None:
                flow_velocities.append(velocity)
            
            # 血流动力学参数
            hemodynamic = result.get('hemodynamic_modeling', {})
            if hemodynamic:
                fvi = hemodynamic.get('flow_velocity_index')
                if fvi is not None:
                    flow_velocity_indices.append(fvi)
                
                coi = hemodynamic.get('cardiac_output_index')
                if coi is not None:
                    cardiac_output_indices.append(coi)
                
                compliance = hemodynamic.get('vascular_compliance')
                if compliance is not None:
                    vascular_compliances.append(compliance)
                
                resistance = hemodynamic.get('vascular_resistance')
                if resistance is not None:
                    vascular_resistances.append(resistance)
        
        # 计算统计值
        statistics = {
            'perfusion_index_stats': self._calculate_stats(perfusion_indices),
            'flow_velocity_stats': self._calculate_stats(flow_velocities),
            'flow_velocity_index_stats': self._calculate_stats(flow_velocity_indices),
            'cardiac_output_index_stats': self._calculate_stats(cardiac_output_indices),
            'vascular_compliance_stats': self._calculate_stats(vascular_compliances),
            'vascular_resistance_stats': self._calculate_stats(vascular_resistances),
            'flow_status_distribution': flow_status_counts,
            'total_analyzed_segments': len(blood_flow_results)
        }
        
        return statistics
    
    def _calculate_stats(self, values):
        """
        计算统计值的辅助函数
        
        Args:
            values: 数值列表
            
        Returns:
            stats: 统计信息字典
        """
        if not values:
            return {'mean': None, 'std': None, 'min': None, 'max': None, 'count': 0}
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'count': len(values)
        }
    
    def assess_blood_flow_risk(self, blood_flow_statistics):
        """
        评估血流相关风险
        
        Args:
            blood_flow_statistics: 血流统计信息
            
        Returns:
            risk_assessment: 血流风险评估
        """
        risk_factors = []
        risk_score = 0
        
        # 灌注指数风险评估
        pi_stats = blood_flow_statistics.get('perfusion_index_stats', {})
        if pi_stats.get('mean') is not None:
            pi_mean = pi_stats['mean']
            if pi_mean < 0.3:
                risk_factors.append("灌注指数过低 (<0.3%)")
                risk_score += 3
            elif pi_mean < 0.5:
                risk_factors.append("灌注指数偏低 (<0.5%)")
                risk_score += 1
            elif pi_mean > 8.0:
                risk_factors.append("灌注指数过高 (>8.0%)")
                risk_score += 2
        
        # 血流状态分布评估
        flow_status_dist = blood_flow_statistics.get('flow_status_distribution', {})
        total_segments = blood_flow_statistics.get('total_analyzed_segments', 1)
        
        low_perfusion_ratio = flow_status_dist.get('低灌注', 0) / total_segments
        if low_perfusion_ratio > 0.5:
            risk_factors.append("低灌注段比例过高 (>50%)")
            risk_score += 3
        elif low_perfusion_ratio > 0.3:
            risk_factors.append("低灌注段比例偏高 (>30%)")
            risk_score += 1
        
        # 血管顺应性评估
        compliance_stats = blood_flow_statistics.get('vascular_compliance_stats', {})
        if compliance_stats.get('mean') is not None:
            compliance_mean = compliance_stats['mean']
            if compliance_mean < 0.8:
                risk_factors.append("血管顺应性降低")
                risk_score += 2
        
        # 血管阻力评估
        resistance_stats = blood_flow_statistics.get('vascular_resistance_stats', {})
        if resistance_stats.get('mean') is not None:
            resistance_mean = resistance_stats['mean']
            if resistance_mean > 2.0:
                risk_factors.append("血管阻力增高")
                risk_score += 2
        
        # 血流速度变异性评估
        velocity_stats = blood_flow_statistics.get('flow_velocity_stats', {})
        if velocity_stats.get('std') is not None:
            velocity_std = velocity_stats['std']
            if velocity_std > 10:  # 变异性过大
                risk_factors.append("血流速度变异性过大")
                risk_score += 1
        
        # 风险等级评估
        if risk_score >= 6:
            risk_level = "高血流风险"
        elif risk_score >= 3:
            risk_level = "中等血流风险"
        elif risk_score >= 1:
            risk_level = "低血流风险"
        else:
            risk_level = "血流正常"
        
        # 生成血流相关建议
        recommendations = self.generate_blood_flow_recommendations(risk_level, risk_factors)
        
        risk_assessment = {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendations': recommendations
        }
        
        return risk_assessment
    
    def generate_blood_flow_recommendations(self, risk_level, risk_factors):
        """
        生成血流相关健康建议
        
        Args:
            risk_level: 血流风险等级
            risk_factors: 血流风险因素
            
        Returns:
            recommendations: 建议列表
        """
        recommendations = []
        
        if risk_level == "高血流风险":
            recommendations.extend([
                "建议立即就医检查血流循环功能",
                "评估是否存在血管狭窄或阻塞",
                "检查心脏泵血功能",
                "考虑血管扩张治疗"
            ])
        elif risk_level == "中等血流风险":
            recommendations.extend([
                "建议定期监测血流灌注状态",
                "适量进行促进血液循环的运动",
                "保持温暖，避免血管收缩",
                "考虑改善微循环的治疗"
            ])
        elif risk_level == "低血流风险":
            recommendations.extend([
                "注意保持良好的血液循环",
                "适量运动促进血流",
                "避免长时间保持同一姿势"
            ])
        else:
            recommendations.extend([
                "血流状态良好，继续保持",
                "定期监测血流健康"
            ])
        
        # 针对特定风险因素的建议
        for factor in risk_factors:
            if "灌注指数" in factor:
                if "过低" in factor:
                    recommendations.append("重点改善末梢血液循环")
                elif "过高" in factor:
                    recommendations.append("注意血管炎症或充血状态")
            if "低灌注" in factor:
                recommendations.append("加强血流灌注改善措施")
            if "血管顺应性" in factor:
                recommendations.append("注意血管弹性维护")
            if "血管阻力" in factor:
                recommendations.append("考虑血管扩张治疗")
            if "变异性" in factor:
                recommendations.append("稳定血流动力学状态")
        
        return list(set(recommendations))  # 去重
    
    def _get_default_sleep_result(self) -> Dict[str, Any]:
        """获取默认的睡眠分析结果"""
        return {
            'sleep_apnea_analysis': {
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
            },
            'nocturnal_spo2_analysis': {
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
            },
            'blood_pressure_rhythm_analysis': {
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
            },
            'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_duration_hours': 0.0
        }

    def process_all_files(self):
        """
        处理所有JSON文件
        """
        if not os.path.exists(self.input_dir):
            print(f"输入目录不存在: {self.input_dir}")
            return
        
        # 获取所有JSON文件
        try:
            with os.scandir(self.input_dir) as it:
                json_files = [entry.name for entry in it if entry.is_file() and entry.name.endswith('.json')]
        except Exception:
            json_files = [f for f in os.listdir(self.input_dir) if f.endswith('.json')]
        
        if not json_files:
            print(f"在目录 {self.input_dir} 中未找到JSON文件")
            return
        
        if os.environ.get('VASCULAR_VERBOSE', '0') == '1':
            print(f"找到 {len(json_files)} 个JSON文件")
        
        processed_count = 0
        failed_count = 0
        file_paths = [os.path.join(self.input_dir, fn) for fn in json_files]

        # 选择合理的并行度，保留一个核心给系统
        env_fw = os.environ.get('VASCULAR_FILE_WORKERS')
        if env_fw and env_fw.isdigit():
            max_workers = max(1, int(env_fw))
        else:
            max_workers = max(1, (os.cpu_count() or 2) - 1)
        max_workers = min(max_workers, len(file_paths))
        if os.environ.get('VASCULAR_VERBOSE', '0') == '1':
            print(f"并行处理进程数: {max_workers}")

        def _worker(file_path: str) -> bool:
            try:
                result = self.process_device_data(file_path)
                if result is not None:
                    self.save_analysis_result(result['device_id'], result)
                    return True
                return False
            except Exception:
                return False

        exec_type = os.environ.get('VASCULAR_FILE_EXECUTOR', 'process')
        if exec_type == 'thread':
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                inflight = set()
                total = len(file_paths)
                env_mi = os.environ.get('VASCULAR_MAX_INFLIGHT')
                max_inflight = max_workers * 4 if not (env_mi and env_mi.isdigit()) else max(max_workers, int(env_mi))
                it = iter(file_paths)
                def _fill():
                    while len(inflight) < max_inflight:
                        try:
                            fp = next(it)
                        except StopIteration:
                            break
                        inflight.add(executor.submit(_process_and_save_entry, fp, self.input_dir, self.output_dir))
                _fill()
                done_count = 0
                while inflight:
                    done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                    for fut in done:
                        inflight.remove(fut)
                        try:
                            success, filename = fut.result()
                            if success:
                                processed_count += 1
                            else:
                                failed_count += 1
                        except Exception:
                            failed_count += 1
                        done_count += 1
                        if os.environ.get('VASCULAR_VERBOSE', '0') == '1':
                            print(f"进度: {done_count}/{total} ({(done_count / total * 100):.1f}%)")
                    _fill()
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                inflight = set()
                total = len(file_paths)
                env_mi = os.environ.get('VASCULAR_MAX_INFLIGHT')
                max_inflight = max_workers * 4 if not (env_mi and env_mi.isdigit()) else max(max_workers, int(env_mi))
                it = iter(file_paths)
                def _fill():
                    while len(inflight) < max_inflight:
                        try:
                            fp = next(it)
                        except StopIteration:
                            break
                        inflight.add(executor.submit(_process_and_save_entry, fp, self.input_dir, self.output_dir))
                _fill()
                done_count = 0
                while inflight:
                    done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                    for fut in done:
                        inflight.remove(fut)
                        try:
                            success, filename = fut.result()
                            if success:
                                processed_count += 1
                            else:
                                failed_count += 1
                        except Exception:
                            failed_count += 1
                        done_count += 1
                        if os.environ.get('VASCULAR_VERBOSE', '0') == '1':
                            print(f"进度: {done_count}/{total} ({(done_count / total * 100):.1f}%)")
                    _fill()

        if os.environ.get('VASCULAR_VERBOSE', '0') == '1':
            print(f"\n处理完成!")
            print(f"成功处理: {processed_count} 个文件")
            print(f"处理失败: {failed_count} 个文件")
            print(f"结果保存在: {self.output_dir}")

    def analyze_hyperglycemia_segment(self, decompressed_data, segment_index, segment_info):
        """
        分析单个数据段的高血糖参数
        
        Args:
            decompressed_data: 解压缩的PPG数据
            segment_index: 数据段索引
            segment_info: 数据段信息
            
        Returns:
            hyperglycemia_result: 高血糖分析结果
        """
        try:
            ppg_signal = decompressed_data if isinstance(decompressed_data, np.ndarray) and decompressed_data.dtype == float else np.array(decompressed_data, dtype=float)
            
            # 高血糖分析
            hyperglycemia_analysis = self.hyperglycemia_analyzer.analyze_hyperglycemia_risk(ppg_signal)
            
            # 组装结果
            result = {
                'segment_index': segment_index,
                'collect_time': segment_info.get('collectTime'),
                'create_time': segment_info.get('createTime'),
                'signal_quality': hyperglycemia_analysis.get('signal_quality', 0.0),
                'morphological_features': hyperglycemia_analysis.get('morphological_features', {}),
                'variability_features': hyperglycemia_analysis.get('variability_features', {}),
                'frequency_features': hyperglycemia_analysis.get('frequency_features', {}),
                'statistical_features': hyperglycemia_analysis.get('statistical_features', {}),
                'risk_score': hyperglycemia_analysis.get('risk_score', 0.0),
                'risk_level': hyperglycemia_analysis.get('risk_level', '低风险'),
                'recommendations': hyperglycemia_analysis.get('recommendations', [])
            }
            
            return result
            
        except Exception as e:
            print(f"高血糖分析失败 (段 {segment_index}): {e}")
            return None

    def analyze_hyperlipidemia_segment(self, decompressed_data, segment_index, segment_info):
        """
        分析单个数据段的高血脂参数
        
        Args:
            decompressed_data: 解压缩的PPG数据
            segment_index: 数据段索引
            segment_info: 数据段信息
            
        Returns:
            hyperlipidemia_result: 高血脂分析结果
        """
        try:
            ppg_signal = decompressed_data if isinstance(decompressed_data, np.ndarray) and decompressed_data.dtype == float else np.array(decompressed_data, dtype=float)
            
            # 高血脂分析
            hyperlipidemia_analysis = self.hyperlipidemia_analyzer.analyze_hyperlipidemia_risk(ppg_signal)
            
            # 组装结果
            result = {
                'segment_index': segment_index,
                'collect_time': segment_info.get('collectTime'),
                'create_time': segment_info.get('createTime'),
                'signal_quality': hyperlipidemia_analysis.get('signal_quality', 0.0),
                'vascular_stiffness_features': hyperlipidemia_analysis.get('vascular_stiffness_features', {}),
                'pulse_wave_features': hyperlipidemia_analysis.get('pulse_wave_features', {}),
                'perfusion_features': hyperlipidemia_analysis.get('perfusion_features', {}),
                'arterial_compliance_features': hyperlipidemia_analysis.get('arterial_compliance_features', {}),
                'risk_score': hyperlipidemia_analysis.get('risk_score', 0.0),
                'risk_level': hyperlipidemia_analysis.get('risk_level', '低风险'),
                'recommendations': hyperlipidemia_analysis.get('recommendations', [])
            }
            
            return result
            
        except Exception as e:
            print(f"高血脂分析失败 (段 {segment_index}): {e}")
            return None

    def calculate_hyperglycemia_statistics(self, hyperglycemia_results):
        """计算高血糖分析统计信息"""
        if not hyperglycemia_results:
            return self._get_default_hyperglycemia_statistics()
        
        try:
            # 提取风险评分
            risk_scores = [result.get('risk_score', 0.0) for result in hyperglycemia_results]
            risk_levels = [result.get('risk_level', '低风险') for result in hyperglycemia_results]
            
            # 计算统计信息
            statistics = {
                'total_segments': len(hyperglycemia_results),
                'risk_score_statistics': {
                    'mean': np.mean(risk_scores) if risk_scores else 0.0,
                    'std': np.std(risk_scores) if risk_scores else 0.0,
                    'min': np.min(risk_scores) if risk_scores else 0.0,
                    'max': np.max(risk_scores) if risk_scores else 0.0,
                    'median': np.median(risk_scores) if risk_scores else 0.0
                },
                'risk_level_distribution': {
                    '低风险': risk_levels.count('低风险'),
                    '中风险': risk_levels.count('中风险'),
                    '高风险': risk_levels.count('高风险')
                },
                'high_risk_percentage': (risk_levels.count('高风险') / len(risk_levels) * 100) if risk_levels else 0.0,
                'average_signal_quality': np.mean([result.get('signal_quality', 0.0) for result in hyperglycemia_results])
            }
            
            return statistics
            
        except Exception as e:
            print(f"计算高血糖统计信息失败: {e}")
            return self._get_default_hyperglycemia_statistics()

    def calculate_hyperlipidemia_statistics(self, hyperlipidemia_results):
        """计算高血脂分析统计信息"""
        if not hyperlipidemia_results:
            return self._get_default_hyperlipidemia_statistics()
        
        try:
            # 提取风险评分
            risk_scores = [result.get('risk_score', 0.0) for result in hyperlipidemia_results]
            risk_levels = [result.get('risk_level', '低风险') for result in hyperlipidemia_results]
            
            # 计算统计信息
            statistics = {
                'total_segments': len(hyperlipidemia_results),
                'risk_score_statistics': {
                    'mean': np.mean(risk_scores) if risk_scores else 0.0,
                    'std': np.std(risk_scores) if risk_scores else 0.0,
                    'min': np.min(risk_scores) if risk_scores else 0.0,
                    'max': np.max(risk_scores) if risk_scores else 0.0,
                    'median': np.median(risk_scores) if risk_scores else 0.0
                },
                'risk_level_distribution': {
                    '低风险': risk_levels.count('低风险'),
                    '中风险': risk_levels.count('中风险'),
                    '高风险': risk_levels.count('高风险')
                },
                'high_risk_percentage': (risk_levels.count('高风险') / len(risk_levels) * 100) if risk_levels else 0.0,
                'average_signal_quality': np.mean([result.get('signal_quality', 0.0) for result in hyperlipidemia_results])
            }
            
            return statistics
            
        except Exception as e:
            print(f"计算高血脂统计信息失败: {e}")
            return self._get_default_hyperlipidemia_statistics()

    def assess_hyperglycemia_risk(self, hyperglycemia_statistics):
        """评估高血糖风险"""
        try:
            mean_risk_score = hyperglycemia_statistics.get('risk_score_statistics', {}).get('mean', 0.0)
            high_risk_percentage = hyperglycemia_statistics.get('high_risk_percentage', 0.0)
            
            # 风险等级判断
            if mean_risk_score >= 0.7 or high_risk_percentage >= 30:
                risk_level = "高风险"
                recommendations = [
                    "建议立即就医进行血糖检测",
                    "监控血糖水平变化",
                    "调整饮食结构，减少糖分摄入",
                    "增加有氧运动",
                    "定期监测PPG信号变化"
                ]
            elif mean_risk_score >= 0.4 or high_risk_percentage >= 15:
                risk_level = "中风险"
                recommendations = [
                    "建议定期检查血糖",
                    "注意饮食控制",
                    "保持适量运动",
                    "监测体重变化"
                ]
            else:
                risk_level = "低风险"
                recommendations = [
                    "保持健康生活方式",
                    "定期体检"
                ]
            
            return {
                'overall_risk_level': risk_level,
                'mean_risk_score': mean_risk_score,
                'high_risk_percentage': high_risk_percentage,
                'recommendations': recommendations
            }
            
        except Exception as e:
            print(f"高血糖风险评估失败: {e}")
            return {
                'overall_risk_level': '未知',
                'mean_risk_score': 0.0,
                'high_risk_percentage': 0.0,
                'recommendations': ['数据不足，无法评估']
            }

    def assess_hyperlipidemia_risk(self, hyperlipidemia_statistics):
        """评估高血脂风险"""
        try:
            mean_risk_score = hyperlipidemia_statistics.get('risk_score_statistics', {}).get('mean', 0.0)
            high_risk_percentage = hyperlipidemia_statistics.get('high_risk_percentage', 0.0)
            
            # 风险等级判断
            if mean_risk_score >= 0.7 or high_risk_percentage >= 30:
                risk_level = "高风险"
                recommendations = [
                    "建议立即就医进行血脂检测",
                    "监控血脂水平变化",
                    "调整饮食结构，减少饱和脂肪摄入",
                    "增加有氧运动",
                    "考虑药物治疗",
                    "定期监测PPG信号变化"
                ]
            elif mean_risk_score >= 0.4 or high_risk_percentage >= 15:
                risk_level = "中风险"
                recommendations = [
                    "建议定期检查血脂",
                    "注意低脂饮食",
                    "保持适量运动",
                    "控制体重"
                ]
            else:
                risk_level = "低风险"
                recommendations = [
                    "保持健康生活方式",
                    "定期体检"
                ]
            
            return {
                'overall_risk_level': risk_level,
                'mean_risk_score': mean_risk_score,
                'high_risk_percentage': high_risk_percentage,
                'recommendations': recommendations
            }
            
        except Exception as e:
            print(f"高血脂风险评估失败: {e}")
            return {
                'overall_risk_level': '未知',
                'mean_risk_score': 0.0,
                'high_risk_percentage': 0.0,
                'recommendations': ['数据不足，无法评估']
            }

    def _get_default_hyperglycemia_statistics(self):
        """获取默认高血糖统计信息"""
        return {
            'total_segments': 0,
            'risk_score_statistics': {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            },
            'risk_level_distribution': {
                '低风险': 0,
                '中风险': 0,
                '高风险': 0
            },
            'high_risk_percentage': 0.0,
            'average_signal_quality': 0.0
        }

    def _get_default_hyperlipidemia_statistics(self):
        """获取默认高血脂统计信息"""
        return {
            'total_segments': 0,
            'risk_score_statistics': {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            },
            'risk_level_distribution': {
                '低风险': 0,
                '中风险': 0,
                '高风险': 0
            },
            'high_risk_percentage': 0.0,
            'average_signal_quality': 0.0
        }

def main():
    """主函数"""
    print("PPG血管功能评价分析系统")
    print("=" * 50)
    
    # 创建处理器
    processor = VascularAnalysisProcessor()
    
    # 处理所有文件
    if os.environ.get('PROFILE_VASCULAR', '1') == '1':
        import cProfile, pstats
        import io
        profiles_dir = os.path.join(os.path.dirname(__file__), 'profiles')
        try:
            os.makedirs(profiles_dir, exist_ok=True)
        except Exception:
            pass
        prof_path = os.path.join(profiles_dir, 'vascular_profile.prof')
        txt_path = os.path.join(profiles_dir, 'vascular_profile.txt')
        pr = cProfile.Profile()
        pr.enable()
        processor.process_all_files()
        pr.disable()
        try:
            pr.dump_stats(prof_path)
        except Exception:
            pass
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(50)
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(s.getvalue())
            print(f"性能分析输出: {txt_path}")
        except Exception:
            print(s.getvalue())
    else:
        processor.process_all_files()

if __name__ == "__main__":
    main()