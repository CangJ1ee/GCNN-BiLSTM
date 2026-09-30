import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, DataLoader as GeoDataLoader
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json
import torch.nn.functional as F
from datetime import datetime
import os

warnings.filterwarnings("ignore", category=UserWarning)


class SecurityAdvisor:
    """基于LLM的安全建议生成器"""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.attack_patterns = {
            'DDoS': {
                'description': 'Distributed Denial of Service attack detected',
                'severity': 'High',
                'characteristics': ['High packet rate', 'Multiple source IPs', 'Target overwhelm']
            },
            'PortScan': {
                'description': 'Port scanning activity detected',
                'severity': 'Medium',
                'characteristics': ['Sequential port access', 'Reconnaissance behavior', 'Systematic probing']
            },
            'Infiltration': {
                'description': 'Infiltration attempt detected',
                'severity': 'High',
                'characteristics': ['Unauthorized access', 'Privilege escalation', 'System compromise']
            },
            'WebAttack': {
                'description': 'Web-based attack detected',
                'severity': 'Medium',
                'characteristics': ['SQL injection', 'XSS attempts', 'Web vulnerability exploitation']
            },
            'BENIGN': {
                'description': 'Normal network traffic',
                'severity': 'Low',
                'characteristics': ['Regular communication patterns', 'Expected protocols', 'Normal data flow']
            }
        }

    def init_llm(self):
        """初始化LLM模型（可选）"""
        try:
            model_name = "microsoft/DialoGPT-small"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("LLM模型加载成功")
        except Exception as e:
            print(f"LLM模型加载失败: {e}")
            print("将使用基于规则的建议生成")

    def analyze_context(self, attack_type, feature_importance, confidence_score):
        """分析攻击上下文"""
        context = {
            'attack_type': attack_type,
            'confidence': confidence_score,
            'timestamp': datetime.now().isoformat(),
            'severity': self.attack_patterns.get(attack_type, {}).get('severity', 'Unknown'),
            'top_features': feature_importance[:5] if feature_importance else [],
        }

        if attack_type in self.attack_patterns:
            context['description'] = self.attack_patterns[attack_type]['description']
            context['characteristics'] = self.attack_patterns[attack_type]['characteristics']

        return context

    def generate_security_advice(self, context):
        """生成安全建议"""
        attack_type = context['attack_type']
        confidence = context['confidence']
        severity = context['severity']

        base_advice = self._get_base_advice(attack_type, severity, confidence)

        if self.model and self.tokenizer:
            try:
                enhanced_advice = self._generate_llm_advice(context)
                return {
                    'base_recommendations': base_advice,
                    'detailed_analysis': enhanced_advice,
                    'context': context
                }
            except Exception as e:
                print(f"LLM建议生成失败: {e}")

        return {
            'base_recommendations': base_advice,
            'context': context
        }

    def _get_base_advice(self, attack_type, severity, confidence):
        """获取基础安全建议"""
        advice_map = {
            'DDoS': [
                "立即启用DDoS防护机制",
                "增加带宽容量或启用流量清洗",
                "配置防火墙规则限制异常流量",
                "监控网络负载并设置告警",
                "考虑使用CDN分散流量压力"
            ],
            'PortScan': [
                "加强端口访问控制",
                "启用入侵检测系统(IDS)",
                "关闭不必要的开放端口",
                "实施网络分段策略",
                "增强日志监控和分析"
            ],
            'Infiltration': [
                "立即隔离受影响的系统",
                "检查系统 integrity",
                "更新所有安全补丁",
                "重置可能泄露的凭据",
                "进行全面的安全审计"
            ],
            'WebAttack': [
                "更新Web应用防火墙(WAF)规则",
                "检查Web应用程序漏洞",
                "实施输入验证和过滤",
                "启用HTTPS和安全头",
                "定期进行渗透测试"
            ],
            'BENIGN': [
                "保持当前安全策略",
                "继续监控网络活动",
                "定期更新安全配置",
                "维护日志记录"
            ]
        }

        base_advice = advice_map.get(attack_type, ["需要进一步分析此攻击类型"])

        if confidence > 0.9:
            urgency_prefix = "【高置信度】紧急处理："
        elif confidence > 0.7:
            urgency_prefix = "【中等置信度】重要处理："
        else:
            urgency_prefix = "【低置信度】谨慎处理："

        return [urgency_prefix] + base_advice

    def _generate_llm_advice(self, context):
        """使用LLM生成详细建议"""
        prompt = f"""
        网络安全分析：
        攻击类型: {context['attack_type']}
        置信度: {context['confidence']:.2f}
        严重程度: {context['severity']}

        请提供专业的安全建议：
        """

        inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()


class SHAPExplainer:
    """SHAP解释器，用于分析特征重要性 - 完全修复版"""

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = list(feature_names)  # 确保是列表
        self.explainer = None
        self.shap_values = None
        self.test_data = None
        self.expected_value = None

    def create_explainer(self, background_data, method='kernel'):
        """创建SHAP解释器"""
        try:
            def model_wrapper(x):
                x_tensor = torch.tensor(x, dtype=torch.float32).to(next(self.model.parameters()).device)
                with torch.no_grad():
                    outputs = self.model(x_tensor, use_graph=False)
                    return F.softmax(outputs, dim=1).cpu().numpy()

            background_sample = background_data[:100] if len(background_data) > 100 else background_data
            self.explainer = shap.KernelExplainer(model_wrapper, background_sample)

            print("SHAP解释器创建成功")
        except Exception as e:
            print(f"SHAP解释器创建失败: {e}")

    def explain_predictions(self, test_data, attack_labels, le, max_samples=750):
        """解释预测结果 - 修复版"""
        try:
            if self.explainer is None:
                print("请先创建SHAP解释器")
                return None

            # 限制样本数量
            if len(test_data) > max_samples:
                indices = np.random.choice(len(test_data), max_samples, replace=False)
                test_sample = test_data[indices]
                labels_sample = attack_labels[indices]
            else:
                test_sample = test_data
                labels_sample = attack_labels

            # 保存测试数据
            self.test_data = np.array(test_sample)
            self.labels_sample = np.array(labels_sample)

            print(f"计算SHAP值，样本数量: {len(test_sample)}")

            # 计算SHAP值
            self.shap_values = self.explainer.shap_values(test_sample)

            # 保存期望值
            if hasattr(self.explainer, 'expected_value'):
                self.expected_value = self.explainer.expected_value

            # 打印SHAP值的结构信息用于调试
            if isinstance(self.shap_values, list):
                print(f"SHAP值结构: 列表，长度={len(self.shap_values)}")
                if len(self.shap_values) > 0:
                    print(f"每个元素形状: {np.array(self.shap_values[0]).shape}")
            else:
                print(f"SHAP值结构: 数组，形状={self.shap_values.shape}")

            # 分析每种攻击类型的特征重要性
            attack_feature_importance = self._analyze_by_attack_type(labels_sample, le)

            return attack_feature_importance

        except Exception as e:
            print(f"SHAP解释失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _analyze_by_attack_type(self, attack_labels, le):
        """按攻击类型分析特征重要性 - 修复版"""
        attack_importance = {}

        if self.shap_values is None:
            return attack_importance

        unique_labels = np.unique(attack_labels)

        for label in unique_labels:
            try:
                label_int = int(label)
                label_mask = attack_labels == label_int

                if isinstance(self.shap_values, list):
                    # 多类分类情况
                    if label_int < len(self.shap_values):
                        class_shap_values = np.array(self.shap_values[label_int])
                        if len(class_shap_values.shape) > 2:
                            class_shap_values = class_shap_values.reshape(class_shap_values.shape[0], -1)
                        label_shap = class_shap_values[label_mask]
                    else:
                        continue
                else:
                    label_shap = self.shap_values[label_mask]

                if len(label_shap) > 0:
                    # 确保是2D数组
                    if len(label_shap.shape) == 1:
                        label_shap = label_shap.reshape(1, -1)

                    # 计算平均SHAP值的绝对值
                    mean_shap = np.abs(label_shap).mean(axis=0)

                    # 确保mean_shap是一维数组
                    mean_shap = mean_shap.flatten()

                    # 获取最重要的特征
                    n_features = min(10, len(mean_shap), len(self.feature_names))
                    top_indices = np.argsort(mean_shap)[-n_features:][::-1]

                    top_features = []
                    for i in top_indices:
                        idx = int(i)
                        if idx < len(self.feature_names):
                            top_features.append((self.feature_names[idx], float(mean_shap[idx])))

                    # 获取攻击类型名称
                    attack_name = le.inverse_transform([label_int])[0]

                    attack_importance[attack_name] = {
                        'label': label_int,
                        'top_features': top_features,
                        'mean_shap_values': mean_shap,
                        'sample_count': int(np.sum(label_mask))
                    }

            except Exception as e:
                print(f"处理标签 {label} 时出错: {e}")
                continue

        return attack_importance

    def _get_aggregated_shap_values(self):
        """获取聚合的SHAP值 - 用于绘图"""
        if self.shap_values is None:
            return None

        if isinstance(self.shap_values, list):
            # 多类分类：取所有类别SHAP值的绝对值平均
            all_shap = []
            for sv in self.shap_values:
                sv_array = np.array(sv)
                if len(sv_array.shape) > 2:
                    sv_array = sv_array.reshape(sv_array.shape[0], -1)
                all_shap.append(sv_array)

            # 确保所有数组形状一致
            min_features = min(s.shape[1] for s in all_shap)
            all_shap = [s[:, :min_features] for s in all_shap]

            # 计算平均
            aggregated = np.mean([np.abs(s) for s in all_shap], axis=0)
            return aggregated
        else:
            return self.shap_values

    def plot_beeswarm(self, class_idx=None, class_name=None, max_display=15, save_path=None):
        """绘制SHAP蜂群图 - 修复版"""
        if self.shap_values is None or self.test_data is None:
            print("请先运行explain_predictions计算SHAP值")
            return

        try:
            # 获取SHAP值
            if isinstance(self.shap_values, list):
                if class_idx is not None and class_idx < len(self.shap_values):
                    shap_vals = np.array(self.shap_values[class_idx])
                else:
                    # 使用第一个类别或平均
                    shap_vals = np.mean([np.abs(np.array(sv)) for sv in self.shap_values], axis=0)
            else:
                shap_vals = np.array(self.shap_values)

            # 确保形状正确
            if len(shap_vals.shape) > 2:
                shap_vals = shap_vals.reshape(shap_vals.shape[0], -1)

            # 确保特征数量匹配
            n_samples = shap_vals.shape[0]
            n_features = min(shap_vals.shape[1], len(self.feature_names), self.test_data.shape[1])

            shap_vals = shap_vals[:, :n_features]
            test_data = self.test_data[:n_samples, :n_features]
            feature_names = self.feature_names[:n_features]

            print(
                f"绘图数据形状 - SHAP值: {shap_vals.shape}, 测试数据: {test_data.shape}, 特征数: {len(feature_names)}")

            # 使用手动方法绘制蜂群图
            self._plot_beeswarm_manual_v2(shap_vals, test_data, feature_names,
                                          class_name, max_display, save_path)

        except Exception as e:
            print(f"绘制蜂群图失败: {e}")
            import traceback
            traceback.print_exc()

    def _plot_beeswarm_manual_v2(self, shap_vals, feature_data, feature_names,
                                 class_name=None, max_display=15, save_path=None):
        """手动实现蜂群图 - 完全修复版"""
        print("使用优化的手动方法绘制蜂群图...")

        try:
            # 确保数据是numpy数组
            shap_vals = np.array(shap_vals)
            feature_data = np.array(feature_data)

            # 验证维度
            if len(shap_vals.shape) != 2:
                print(f"SHAP值维度错误: {shap_vals.shape}")
                return

            n_samples, n_features = shap_vals.shape

            # 确保特征数据维度匹配
            if feature_data.shape[0] != n_samples:
                print(f"样本数不匹配: SHAP={n_samples}, 特征={feature_data.shape[0]}")
                return

            if feature_data.shape[1] != n_features:
                # 截断到相同维度
                min_features = min(feature_data.shape[1], n_features)
                shap_vals = shap_vals[:, :min_features]
                feature_data = feature_data[:, :min_features]
                n_features = min_features
                feature_names = feature_names[:min_features]

            # 计算每个特征的平均绝对SHAP值
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)

            # 获取top特征的索引
            n_display = min(max_display, n_features)
            feature_order = np.argsort(mean_abs_shap)[-n_display:]

            fig, ax = plt.subplots(figsize=(12, max(6, n_display * 0.5)))

            # 为每个特征绘制散点
            for i, feat_idx in enumerate(feature_order):
                feat_shap = shap_vals[:, feat_idx]
                feat_values = feature_data[:, feat_idx]

                # 归一化特征值用于颜色映射
                feat_min, feat_max = feat_values.min(), feat_values.max()
                if feat_max - feat_min > 1e-10:
                    normalized_values = (feat_values - feat_min) / (feat_max - feat_min)
                else:
                    normalized_values = np.ones_like(feat_values) * 0.5

                # 添加抖动
                y_jitter = np.random.normal(i, 0.15, size=len(feat_shap))

                # 绘制散点
                scatter = ax.scatter(feat_shap, y_jitter, c=normalized_values,
                                     cmap='coolwarm', alpha=0.6, s=15,
                                     vmin=0, vmax=1)

            # 设置y轴标签
            ax.set_yticks(range(len(feature_order)))
            y_labels = []
            for idx in feature_order:
                fname = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
                if len(fname) > 30:
                    fname = fname[:27] + '...'
                y_labels.append(fname)
            ax.set_yticklabels(y_labels)

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label('Feature value', rotation=270, labelpad=15)
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels(['Low', 'Medium', 'High'])

            # 添加零线
            ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

            ax.set_xlabel('SHAP value (impact on model output)', fontsize=11)

            if class_name:
                ax.set_title(f'SHAP Feature Importance - {class_name}', fontsize=13)
            else:
                ax.set_title('SHAP Feature Importance (Beeswarm Plot)', fontsize=13)

            ax.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"图片已保存至: {save_path}")

            plt.show()

        except Exception as e:
            print(f"手动绘制蜂群图失败: {e}")
            import traceback
            traceback.print_exc()

    def plot_beeswarm_for_all_classes(self, le, max_display=15, save_dir='shap_plots'):
        """为所有攻击类型绘制蜂群图"""
        if self.shap_values is None:
            print("请先运行explain_predictions计算SHAP值")
            return

        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if isinstance(self.shap_values, list):
            n_classes = len(self.shap_values)
            for class_idx in range(n_classes):
                try:
                    class_name = le.inverse_transform([class_idx])[0]
                    # 清理文件名中的特殊字符
                    safe_name = class_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                    safe_name = safe_name.replace("\x96", "-").replace("–", "-")
                    save_path = os.path.join(save_dir, f'shap_beeswarm_{safe_name}.png')

                    print(f"\n绘制 {class_name} 的SHAP蜂群图...")
                    self.plot_beeswarm(class_idx=class_idx, class_name=class_name,
                                       max_display=max_display, save_path=save_path)
                except Exception as e:
                    print(f"绘制类别 {class_idx} 失败: {e}")
        else:
            save_path = os.path.join(save_dir, 'shap_beeswarm_overall.png')
            self.plot_beeswarm(max_display=max_display, save_path=save_path)

    def plot_summary_bar(self, max_display=20, save_path=None):
        """绘制SHAP特征重要性条形图 - 修复版"""
        if self.shap_values is None:
            print("请先运行explain_predictions计算SHAP值")
            return

        try:
            # 计算全局特征重要性
            if isinstance(self.shap_values, list):
                # 多类分类：合并所有类别的SHAP值
                all_shap_abs = []
                for sv in self.shap_values:
                    sv_array = np.array(sv)
                    if len(sv_array.shape) > 2:
                        sv_array = sv_array.reshape(sv_array.shape[0], -1)
                    all_shap_abs.append(np.abs(sv_array))

                # 确保所有数组形状一致
                min_features = min(s.shape[1] for s in all_shap_abs)
                all_shap_abs = [s[:, :min_features] for s in all_shap_abs]

                # 堆叠并计算平均
                stacked = np.concatenate(all_shap_abs, axis=0)
                mean_shap = stacked.mean(axis=0)
            else:
                shap_array = np.array(self.shap_values)
                if len(shap_array.shape) > 2:
                    shap_array = shap_array.reshape(shap_array.shape[0], -1)
                mean_shap = np.abs(shap_array).mean(axis=0)

            mean_shap = mean_shap.flatten()

            # 确保特征名数量匹配
            n_features = min(len(mean_shap), len(self.feature_names))
            mean_shap = mean_shap[:n_features]
            feature_names = self.feature_names[:n_features]

            # 获取top特征
            n_display = min(max_display, n_features)
            top_indices = np.argsort(mean_shap)[-n_display:][::-1]

            top_features = [feature_names[i] for i in top_indices]
            top_values = [mean_shap[i] for i in top_indices]

            # 创建图形
            fig, ax = plt.subplots(figsize=(12, max(6, n_display * 0.4)))

            # 绘制条形图
            colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(top_features)))
            y_pos = range(len(top_features))

            # 反转顺序使最重要的在上面
            bars = ax.barh(y_pos, top_values[::-1], color=colors[::-1], edgecolor='navy', alpha=0.8)

            # 设置标签
            y_labels = []
            for f in top_features[::-1]:
                if len(f) > 35:
                    f = f[:32] + '...'
                y_labels.append(f)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(y_labels)
            ax.set_xlabel('mean(|SHAP value|)', fontsize=11)
            ax.set_title('Global Feature Importance (SHAP)', fontsize=13)

            # 添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + max(top_values) * 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{width:.4f}', ha='left', va='center', fontsize=8)

            ax.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"图片已保存至: {save_path}")

            plt.show()

        except Exception as e:
            print(f"绘制特征重要性条形图失败: {e}")
            import traceback
            traceback.print_exc()

    def plot_feature_importance(self, attack_importance, le):
        """绘制每种攻击类型的特征重要性图"""
        if not attack_importance:
            print("没有可用的特征重要性数据")
            return

        n_attacks = len(attack_importance)
        if n_attacks == 0:
            return

        cols = min(2, n_attacks)
        rows = (n_attacks + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(14 * cols, 6 * rows))

        # 确保axes是可迭代的列表
        if n_attacks == 1:
            axes = [axes]
        elif rows == 1 and cols > 1:
            axes = list(axes)
        elif rows > 1 and cols > 1:
            axes = axes.ravel().tolist()
        else:
            axes = [axes]

        plot_count = 0
        for attack_name, importance_data in attack_importance.items():
            if plot_count >= len(axes):
                break

            try:
                top_features = importance_data['top_features'][:10]

                if len(top_features) > 0:
                    features, values = zip(*top_features)
                    features = list(features)
                    values = list(values)

                    ax = axes[plot_count]

                    # 反转顺序
                    y_pos = range(len(features))
                    colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(features)))
                    bars = ax.barh(y_pos, values[::-1], color=colors[::-1], alpha=0.8)

                    # 设置标签
                    y_labels = []
                    for f in features[::-1]:
                        if len(f) > 25:
                            f = f[:22] + '...'
                        y_labels.append(f)

                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(y_labels)
                    ax.set_xlabel('SHAP Value (Average Absolute)')
                    ax.set_title(f'{attack_name} - Top Features (n={importance_data["sample_count"]})')
                    ax.grid(True, axis='x', alpha=0.3)

                    # 添加数值标签
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                                f'{width:.4f}', ha='left', va='center', fontsize=8)

                    plot_count += 1
            except Exception as e:
                print(f"绘制攻击类型 {attack_name} 的特征重要性时出错: {e}")
                continue

        # 隐藏未使用的子图
        for i in range(plot_count, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig('attack_feature_importance.png', dpi=300, bbox_inches='tight')
        print("攻击类型特征重要性图已保存至: attack_feature_importance.png")
        plt.show()


def load_data(dataset_paths):
    """加载并合并多个数据集文件"""
    all_data = []
    for path in dataset_paths:
        try:
            df = pd.read_csv(path, encoding='latin1', low_memory=False)
            all_data.append(df)
            print(f"成功加载: {path}")
        except Exception as e:
            print(f"加载失败 {path}: {e}")

    if not all_data:
        raise ValueError("未能成功加载任何数据集")

    return pd.concat(all_data, ignore_index=True)


def preprocess_data(df_combined):
    """数据预处理"""
    df_combined.columns = df_combined.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')

    if 'Label' not in df_combined.columns:
        raise ValueError("数据集中未找到 'Label' 列")

    df_combined = df_combined.dropna(subset=['Label'])
    df_combined = df_combined[df_combined['Label'].astype(str).str.strip() != '']

    labels = df_combined['Label']
    features_df = df_combined.drop(columns=['Label'])

    cols_to_drop = ['Flow_ID', 'Source_IP', 'Destination_IP', 'Timestamp', 'SimillarHTTP']
    cols_to_drop = [col for col in cols_to_drop if col in features_df.columns]
    features_df = features_df.drop(columns=cols_to_drop, errors='ignore')

    for col in features_df.columns:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(features_df.median())
    features_df = features_df.clip(lower=-1e10, upper=1e10)

    if features_df.isna().any().any() or np.isinf(features_df.values).any():
        raise ValueError("数据预处理后仍包含 NaN 或无穷大值")

    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels.astype(str))
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df.values)

    return scaled_features, encoded_labels, le, features_df.columns.tolist()


def create_similarity_graph(features, threshold=0.8, max_connections=50):
    print(f"创建相似度图，样本数量: {features.shape[0]}")

    n_samples = features.shape[0]

    if n_samples > 20000:
        sample_indices = np.random.choice(n_samples, size=min(20000, n_samples), replace=False)
        features_sampled = features[sample_indices]
        print(f"大数据集采样至: {len(sample_indices)} 个样本")
    else:
        features_sampled = features
        sample_indices = np.arange(n_samples)

    similarity_matrix = cosine_similarity(features_sampled)

    edge_list = []
    for i in range(len(features_sampled)):
        similar_nodes = np.where(similarity_matrix[i] > threshold)[0]
        if len(similar_nodes) > max_connections:
            top_indices = np.argsort(similarity_matrix[i])[-max_connections:]
            similar_nodes = top_indices[similarity_matrix[i][top_indices] > threshold]

        for j in similar_nodes:
            if i != j:
                edge_list.append([i, j])

    if len(edge_list) == 0:
        edge_list = [[i, (i + 1) % len(features_sampled)] for i in range(len(features_sampled))]
        print("警告: 未找到满足阈值的边，创建环形图")

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    print(f"创建图完成，边数: {edge_index.shape[1]}")

    return edge_index, sample_indices


def create_graph_data_fast(features, labels, threshold=0.8):
    """快速创建图数据"""
    edge_index, sample_indices = create_similarity_graph(features, threshold)

    features_sampled = features[sample_indices] if len(sample_indices) < len(features) else features
    labels_sampled = labels[sample_indices] if len(sample_indices) < len(labels) else labels

    x = torch.tensor(features_sampled, dtype=torch.float32)
    y = torch.tensor(labels_sampled, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


class GCNLayer(nn.Module):
    """GCN层实现"""

    def __init__(self, in_features, out_features, dropout=0.1):
        super(GCNLayer, self).__init__()
        self.gcn_conv = GCNConv(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.gcn_conv(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        return x


class GATLayer(nn.Module):
    """GAT层实现"""

    def __init__(self, in_features, out_features, heads=8, dropout=0.1):
        super(GATLayer, self).__init__()
        self.gat_conv = GATConv(in_features, out_features, heads=heads, dropout=dropout, concat=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.gat_conv(x, edge_index)
        x = self.dropout(x)
        return x


class CNNBiLSTM(nn.Module):
    """CNN-BiLSTM组件，动态适应输入维度"""

    def __init__(self, input_size, cnn_out_channels=64, lstm_hidden_size=128):
        super(CNNBiLSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_out_channels, cnn_out_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels * 2),
            nn.MaxPool1d(2)
        )

        self.input_size = input_size
        self.cnn_out_channels = cnn_out_channels
        self.lstm_hidden_size = lstm_hidden_size

        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_size)
            cnn_out = self.cnn(dummy_input)
            cnn_output_size = cnn_out.shape[1] * cnn_out.shape[2]

        self.lstm = nn.LSTM(cnn_output_size, lstm_hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)

        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(batch_size, -1)

        lstm_out, _ = self.lstm(cnn_out.unsqueeze(1))
        return lstm_out[:, -1, :]


class HybridGCNGATCNNBiLSTM(nn.Module):
    """混合GCN-GAT-CNN-BiLSTM模型"""

    def __init__(self, num_features, num_classes, gcn_hidden=128, gat_hidden=128, gat_heads=4,
                 cnn_channels=64, lstm_hidden=128, dropout=0.1):
        super(HybridGCNGATCNNBiLSTM, self).__init__()

        self.gcn1 = GCNLayer(num_features, gcn_hidden, dropout=dropout)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_hidden, dropout=dropout)

        self.gat1 = GATLayer(gcn_hidden, gat_hidden // gat_heads, heads=gat_heads, dropout=dropout)
        self.gat2 = GATLayer(gat_hidden, gat_hidden // gat_heads, heads=gat_heads, dropout=dropout)

        self.cnn_bilstm = CNNBiLSTM(gat_hidden, cnn_channels, lstm_hidden)
        self.cnn_bilstm_raw = CNNBiLSTM(num_features, cnn_channels, lstm_hidden)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, edge_index=None, use_graph=True):
        if use_graph and edge_index is not None:
            x = self.gcn1(x, edge_index)
            x = self.gcn2(x, edge_index)
            x = self.gat1(x, edge_index)
            x = torch.relu(x)
            x = self.gat2(x, edge_index)
            x = torch.relu(x)
            x = self.cnn_bilstm(x)
        else:
            x = self.cnn_bilstm_raw(x)

        x = self.classifier(x)
        return x


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=15):
    """训练模型"""
    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            try:
                if hasattr(batch, 'x'):
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch.x, batch.edge_index)
                    loss = criterion(outputs, batch.y)
                else:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()

            except Exception as e:
                print(f"训练批次 {batch_idx} 出错: {e}")
                continue

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return train_losses


def evaluate_model(model, test_loader, device, le):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            try:
                if hasattr(batch, 'x'):
                    batch = batch.to(device)
                    outputs = model(batch.x, batch.edge_index)
                    labels = batch.y
                else:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)

                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

            except Exception as e:
                print(f"评估批次出错: {e}")
                continue

    if len(all_preds) == 0:
        print("警告: 没有有效的预测结果")
        return 0, 0, 0, 0, [], []

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return accuracy, precision, recall, f1, all_preds, all_probs


def detect_and_advise(model, data_sample, feature_names, le, security_advisor, confidence_threshold=0.7):
    """检测攻击并生成安全建议"""
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        if not isinstance(data_sample, torch.Tensor):
            data_sample = torch.tensor(data_sample, dtype=torch.float32)

        data_sample = data_sample.to(device)
        if len(data_sample.shape) == 1:
            data_sample = data_sample.unsqueeze(0)

        outputs = model(data_sample, use_graph=False)
        probs = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1)
        confidence = torch.max(probs, dim=1)[0]

        attack_type = le.inverse_transform([predicted_class.cpu().item()])[0]
        confidence_score = confidence.cpu().item()

        print(f"\n=== 攻击检测结果 ===")
        print(f"检测到的攻击类型: {attack_type}")
        print(f"置信度: {confidence_score:.4f}")

        if confidence_score >= confidence_threshold:
            feature_importance = []
            data_abs = torch.abs(data_sample[0]).cpu().numpy()
            top_indices = np.argsort(data_abs)[-10:][::-1]
            for idx in top_indices:
                if idx < len(feature_names):
                    feature_importance.append((feature_names[idx], float(data_abs[idx])))

            context = security_advisor.analyze_context(attack_type, feature_importance, confidence_score)
            advice = security_advisor.generate_security_advice(context)

            print(f"\n=== 安全建议 ===")
            print(f"攻击描述: {context.get('description', 'Unknown attack')}")
            print(f"严重程度: {context.get('severity', 'Unknown')}")
            print(f"检测时间: {context.get('timestamp', 'Unknown')}")

            print(f"\n基础安全建议:")
            for i, recommendation in enumerate(advice['base_recommendations'], 1):
                print(f"{i}. {recommendation}")

            if 'detailed_analysis' in advice:
                print(f"\n详细分析:")
                print(advice['detailed_analysis'])

            print(f"\n关键特征:")
            for feature, value in feature_importance[:5]:
                print(f"- {feature}: {value:.4f}")

            return advice
        else:
            print(f"置信度较低 ({confidence_score:.4f} < {confidence_threshold})，建议进一步验证")
            return None


def run_shap_analysis(model, X_test, y_test, feature_names, le, max_samples=750):
    """运行SHAP分析 - 修复版"""
    print("\n=== 开始SHAP特征重要性分析 ===")

    # 创建SHAP解释器
    shap_explainer = SHAPExplainer(model, feature_names)

    # 准备背景数据
    background_data = X_test[:50].numpy() if len(X_test) > 50 else X_test.numpy()
    shap_explainer.create_explainer(background_data)

    # 解释预测
    attack_importance = shap_explainer.explain_predictions(
        X_test[:max_samples].numpy(),
        y_test[:max_samples].numpy(),
        le,
        max_samples=max_samples
    )

    if attack_importance:
        print("\n=== SHAP分析结果 ===")
        for attack_name, importance_data in attack_importance.items():
            print(f"\n{attack_name} 攻击类型的关键特征 (样本数: {importance_data['sample_count']}):")
            for i, (feature, importance) in enumerate(importance_data['top_features'][:5], 1):
                print(f"{i}. {feature}: {importance:.4f}")

        # 绘制特征重要性条形图
        print("\n绘制全局特征重要性条形图...")
        shap_explainer.plot_summary_bar(max_display=20, save_path='shap_summary_bar.png')

        # 绘制蜂群图（全局）
        print("\n绘制全局SHAP蜂群图...")
        shap_explainer.plot_beeswarm(max_display=15, save_path='shap_beeswarm_global.png')

        # 为每个攻击类型绘制蜂群图
        print("\n为每个攻击类型绘制SHAP蜂群图...")
        shap_explainer.plot_beeswarm_for_all_classes(le, max_display=15, save_dir='shap_plots')

        # 绘制按攻击类型分组的特征重要性
        shap_explainer.plot_feature_importance(attack_importance, le)

        return attack_importance, shap_explainer
    else:
        print("SHAP分析失败")
        return None, None


def generate_security_report(model, test_data, test_labels, feature_names, le, security_advisor):
    """生成安全报告"""
    print("\n=== 生成安全报告 ===")

    model.eval()
    device = next(model.parameters()).device

    attack_stats = {}
    sample_advices = {}

    sample_indices = np.random.choice(len(test_data), min(20, len(test_data)), replace=False)

    for idx in sample_indices:
        sample = test_data[idx:idx + 1]
        true_label = test_labels[idx].item()
        true_attack = le.inverse_transform([true_label])[0]

        advice = detect_and_advise(model, sample, feature_names, le, security_advisor, confidence_threshold=0.5)

        if true_attack not in attack_stats:
            attack_stats[true_attack] = {'count': 0, 'detected': 0}

        attack_stats[true_attack]['count'] += 1

        if advice:
            attack_stats[true_attack]['detected'] += 1
            if true_attack not in sample_advices:
                sample_advices[true_attack] = advice

    print(f"\n=== 安全检测报告 ===")
    print(f"检测样本总数: {len(sample_indices)}")
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

    for attack_type, stats in attack_stats.items():
        detection_rate = stats['detected'] / stats['count'] * 100 if stats['count'] > 0 else 0
        print(f"{attack_type}:")
        print(f"  样本数量: {stats['count']}")
        print(f"  检测数量: {stats['detected']}")
        print(f"  检测率: {detection_rate:.1f}%")
        print()

    report_data = {
        'timestamp': datetime.now().isoformat(),
        'attack_statistics': attack_stats,
        'sample_advices': {k: str(v) for k, v in sample_advices.items()}
    }

    try:
        with open('security_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
        print("安全报告已保存到 security_report.json")
    except Exception as e:
        print(f"保存报告失败: {e}")

    return report_data


def main():
    """主函数"""
    dataset_paths = [
        "E:/NSLKdd/SHAP/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "E:/NSLKdd/SHAP/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "E:/NSLKdd/SHAP/TrafficLabelling/Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "E:/NSLKdd/SHAP/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv",
        "E:/NSLKdd/SHAP/TrafficLabelling/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "E:/NSLKdd/SHAP/TrafficLabelling/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "E:/NSLKdd/SHAP/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv",
        "E:/NSLKdd/SHAP/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv"
    ]

    print("=== 1. 数据加载 ===")
    df_combined = load_data(dataset_paths)

    print("=== 2. 数据预处理 ===")
    features, labels, le, feature_names = preprocess_data(df_combined)

    print("=== 3. 数据划分 ===")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    print("=== 4. 创建图数据 ===")
    train_graph = create_graph_data_fast(X_train.numpy(), y_train.numpy())
    test_graph = create_graph_data_fast(X_test.numpy(), y_test.numpy())

    train_loader = GeoDataLoader([train_graph], batch_size=16, shuffle=True)
    test_loader = GeoDataLoader([test_graph], batch_size=16, shuffle=False)

    print("=== 5. 模型初始化 ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    num_features = X_train.shape[1]
    num_classes = len(np.unique(labels))
    print(f"特征数量: {num_features}")
    print(f"类别数量: {num_classes}")
    print(f"攻击类型: {list(le.classes_)}")

    model = HybridGCNGATCNNBiLSTM(num_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("=== 6. 初始化安全顾问 ===")
    security_advisor = SecurityAdvisor()

    print("=== 7. 模型训练 ===")
    train_losses = train_model(model, train_loader, criterion, optimizer, device, num_epochs=100)

    print("=== 8. 模型评估 ===")
    accuracy, precision, recall, f1, predictions, probabilities = evaluate_model(model, test_loader, device, le)

    print("=== 9. SHAP特征重要性分析 ===")
    attack_importance, shap_explainer = run_shap_analysis(model, X_test, y_test, feature_names, le, max_samples=750)

    # 如果SHAP分析成功，生成额外的可视化
    if shap_explainer is not None:
        print("\n=== 生成额外的SHAP可视化 ===")

        # 为特定攻击类型生成蜂群图
        attack_types_to_plot = ['DDoS', 'PortScan', 'BENIGN']
        for attack_name in attack_types_to_plot:
            if attack_name in le.classes_:
                class_idx = list(le.classes_).index(attack_name)
                print(f"\n生成 {attack_name} 的详细SHAP蜂群图...")
                try:
                    safe_name = attack_name.replace(" ", "_").replace("/", "_")
                    shap_explainer.plot_beeswarm(
                        class_idx=class_idx,
                        class_name=attack_name,
                        max_display=15,
                        save_path=f'shap_beeswarm_{safe_name}.png'
                    )
                except Exception as e:
                    print(f"生成 {attack_name} 蜂群图失败: {e}")

    print("=== 10. 实时攻击检测演示 ===")
    # 选择不同类型的测试样本进行演示
    unique_labels = np.unique(y_test.numpy())
    demo_samples = []

    for label in unique_labels[:5]:
        label_indices = np.where(y_test.numpy() == label)[0]
        if len(label_indices) > 0:
            demo_samples.append(label_indices[0])

    for i, idx in enumerate(demo_samples):
        print(f"\n--- 演示样本 {i + 1} ---")
        sample = X_test[idx]
        true_label = y_test[idx].item()
        true_attack = le.inverse_transform([true_label])[0]
        print(f"真实攻击类型: {true_attack}")

        # 检测并生成建议
        advice = detect_and_advise(model, sample, feature_names, le, security_advisor)

    print("=== 11. 生成综合安全报告 ===")
    security_report = generate_security_report(model, X_test, y_test, feature_names, le, security_advisor)

    print("=== 12. 训练损失可视化 ===")
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.title('Training Loss Over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("=== 13. 生成SHAP分析总结报告 ===")
    if attack_importance:
        # 创建SHAP分析总结
        shap_summary = {
            'analysis_time': datetime.now().isoformat(),
            'total_samples_analyzed': min(50, len(X_test)),
            'attack_types_analyzed': list(attack_importance.keys()),
            'feature_importance_by_attack': {}
        }

        for attack_name, data in attack_importance.items():
            shap_summary['feature_importance_by_attack'][attack_name] = {
                'sample_count': data['sample_count'],
                'top_5_features': [
                    {'feature': feat, 'importance': float(imp)}
                    for feat, imp in data['top_features'][:5]
                ]
            }

        # 保存SHAP分析报告
        try:
            with open('shap_analysis_report.json', 'w', encoding='utf-8') as f:
                json.dump(shap_summary, f, ensure_ascii=False, indent=2)
            print("SHAP分析报告已保存到 shap_analysis_report.json")
        except Exception as e:
            print(f"保存SHAP报告失败: {e}")

        # 打印SHAP分析总结
        print("\n" + "=" * 60)
        print("SHAP特征重要性分析总结")
        print("=" * 60)

        for attack_name, data in attack_importance.items():
            print(f"\n【{attack_name}】(样本数: {data['sample_count']})")
            print("-" * 40)
            for i, (feature, importance) in enumerate(data['top_features'][:5], 1):
                print(f"  {i}. {feature}: {importance:.4f}")

    print("\n" + "=" * 60)
    print("=== 系统运行完成 ===")
    print("=" * 60)
    print(f"\n最终模型性能:")
    print(f"  - 准确率 (Accuracy):  {accuracy:.4f}")
    print(f"  - 精确率 (Precision): {precision:.4f}")
    print(f"  - 召回率 (Recall):    {recall:.4f}")
    print(f"  - F1分数 (F1-Score):  {f1:.4f}")

    if attack_importance:
        print(f"\nSHAP分析成功，分析了 {len(attack_importance)} 种攻击类型")
        print(f"SHAP可视化图片保存在当前目录和 shap_plots 文件夹中")

    print(f"\n生成的文件:")
    print(f"  - security_report.json: 安全检测报告")
    print(f"  - shap_analysis_report.json: SHAP分析报告")
    print(f"  - shap_summary_bar.png: 全局特征重要性条形图")
    print(f"  - shap_beeswarm_global.png: 全局SHAP蜂群图")
    print(f"  - training_loss.png: 训练损失曲线")
    print(f"  - confusion_matrix.png: 混淆矩阵")
    print(f"  - attack_feature_importance.png: 攻击类型特征重要性图")
    print(f"  - shap_plots/: 各攻击类型的SHAP蜂群图")

    # 返回结果供进一步分析
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'attack_importance': attack_importance,
        'shap_explainer': shap_explainer,
        'le': le,
        'feature_names': feature_names
    }


if __name__ == "__main__":
    results = main()