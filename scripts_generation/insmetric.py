import os
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Any
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity

class StrokeStructureConsistency:
    """
    笔画级结构一致性度量
    用于评估生成草图与真实草图在局部结构特征上的相似性
    """
    def __init__(self, config: Dict = None):
        """
        初始化评估器
        
        Args:
            config: 配置参数，可设置特征权重等
        """
        if config is None:
            config = {
                'feature_weights': [0.4, 0.3, 0.3],  # 笔画长度、曲率、方向的权重
                'num_direction_bins': 8,
                'num_curvature_bins': 8,
                'num_length_bins': 10
            }
        
        self.config = config
        self.feature_types = ['stroke_length', 'curvature', 'direction', 'spatial_distribution']
        
    def _extract_stroke_features(self, sketch_sequence: np.ndarray) -> Dict[str, np.ndarray]:
        """
        从草图序列中提取笔画级特征
        
        Args:
            sketch_sequence: 草图序列 [N, 3]，每行是 (x, y, pen_state)
                            pen_state: 0-笔落下，1-笔抬起
        
        Returns:
            特征字典
        """
        features = {}
        
        # 1. 笔画长度特征
        stroke_lengths = self._extract_stroke_lengths(sketch_sequence)
        features['stroke_length_dist'] = stroke_lengths
        
        # 2. 曲率特征
        curvatures = self._extract_curvature_features(sketch_sequence)
        features['curvature_dist'] = curvatures
        
        # 3. 方向分布特征
        direction_hist = self._extract_direction_features(sketch_sequence)
        features['direction_hist'] = direction_hist
        
        # 4. 空间分布特征（可选）
        spatial_features = self._extract_spatial_features(sketch_sequence)
        features.update(spatial_features)
        
        return features
    
    def _extract_stroke_lengths(self, sketch_sequence: np.ndarray) -> np.ndarray:
        """提取笔画长度特征"""
        stroke_lengths = []
        current_stroke_len = 0
        in_stroke = False
        
        for i in range(len(sketch_sequence)):
            if sketch_sequence[i, 2] == 0:  # 笔落下
                if not in_stroke:
                    in_stroke = True
                    current_stroke_len = 0
                
                # 计算当前点到下一点的距离
                if i < len(sketch_sequence) - 1 and sketch_sequence[i+1, 2] == 0:
                    dx = sketch_sequence[i+1, 0] - sketch_sequence[i, 0]
                    dy = sketch_sequence[i+1, 1] - sketch_sequence[i, 1]
                    current_stroke_len += np.sqrt(dx**2 + dy**2)
            else:  # 笔抬起
                if in_stroke and current_stroke_len > 0:
                    stroke_lengths.append(current_stroke_len)
                in_stroke = False
        
        # 处理最后一个笔画
        if in_stroke and current_stroke_len > 0:
            stroke_lengths.append(current_stroke_len)
        
        return np.array(stroke_lengths) if stroke_lengths else np.array([0.0])
    
    def _extract_curvature_features(self, sketch_sequence: np.ndarray) -> np.ndarray:
        """提取曲率特征"""
        curvatures = []
        
        for i in range(1, len(sketch_sequence) - 1):
            if sketch_sequence[i, 2] == 0:  # 笔落下时
                p1 = sketch_sequence[i-1, :2]
                p2 = sketch_sequence[i, :2]
                p3 = sketch_sequence[i+1, :2]
                
                # 计算三点形成的角度
                v1 = p1 - p2
                v2 = p3 - p2
                
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                
                if norm1 > 1e-6 and norm2 > 1e-6:
                    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    curvatures.append(angle)
        
        return np.array(curvatures) if curvatures else np.array([0.0])
    
    def _extract_direction_features(self, sketch_sequence: np.ndarray) -> np.ndarray:
        """提取方向分布特征"""
        directions = []
        
        for i in range(len(sketch_sequence) - 1):
            if sketch_sequence[i, 2] == 0 and sketch_sequence[i+1, 2] == 0:
                dx = sketch_sequence[i+1, 0] - sketch_sequence[i, 0]
                dy = sketch_sequence[i+1, 1] - sketch_sequence[i, 1]
                
                # 计算方向角度
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    angle = np.arctan2(dy, dx)
                    directions.append(angle)
        
        # 将角度转换为直方图
        if directions:
            dir_hist, _ = np.histogram(
                directions, 
                bins=self.config['num_direction_bins'], 
                range=(-np.pi, np.pi)
            )
            # 归一化
            dir_hist = dir_hist.astype(float)
            if np.sum(dir_hist) > 0:
                dir_hist = dir_hist / np.sum(dir_hist)
            return dir_hist
        else:
            return np.zeros(self.config['num_direction_bins'])
    
    def _extract_spatial_features(self, sketch_sequence: np.ndarray) -> Dict[str, np.ndarray]:
        """提取空间分布特征"""
        features = {}
        
        if len(sketch_sequence) == 0:
            features['bbox'] = np.array([0, 0, 0, 0])  # xmin, ymin, xmax, ymax
            features['center_of_mass'] = np.array([0, 0])
            features['stroke_count'] = 0
            return features
        
        # 提取笔画点（笔落下状态的点）
        drawing_points = sketch_sequence[sketch_sequence[:, 2] == 0, :2]
        
        if len(drawing_points) > 0:
            # 边界框
            x_min, y_min = np.min(drawing_points, axis=0)
            x_max, y_max = np.max(drawing_points, axis=0)
            features['bbox'] = np.array([x_min, y_min, x_max, y_max])
            
            # 质心
            features['center_of_mass'] = np.mean(drawing_points, axis=0)
            
            # 笔画数量
            stroke_starts = np.where(
                (sketch_sequence[1:, 2] == 0) & (sketch_sequence[:-1, 2] == 1)
            )[0] + 1
            
            if sketch_sequence[0, 2] == 0:
                stroke_starts = np.concatenate([[0], stroke_starts])
            
            features['stroke_count'] = len(stroke_starts)
        else:
            features['bbox'] = np.array([0, 0, 0, 0])
            features['center_of_mass'] = np.array([0, 0])
            features['stroke_count'] = 0
        
        return features
    
    def _compare_length_distributions(self, real_lengths: np.ndarray, gen_lengths: np.ndarray) -> float:
        """比较笔画长度分布"""
        # 创建直方图
        max_length = max(np.max(real_lengths), np.max(gen_lengths), 1.0)
        real_hist, _ = np.histogram(
            real_lengths, 
            bins=self.config['num_length_bins'], 
            range=(0, max_length)
        )
        gen_hist, _ = np.histogram(
            gen_lengths, 
            bins=self.config['num_length_bins'], 
            range=(0, max_length)
        )
        
        # 归一化
        real_hist = real_hist.astype(float)
        gen_hist = gen_hist.astype(float)
        
        if np.sum(real_hist) > 0:
            real_hist = real_hist / np.sum(real_hist)
        if np.sum(gen_hist) > 0:
            gen_hist = gen_hist / np.sum(gen_hist)
        
        # 计算Earth Mover's Distance（简化版）
        emd = np.sum(np.abs(np.cumsum(real_hist) - np.cumsum(gen_hist)))
        
        # 将EMD转换为相似度分数
        # EMD越小，相似度越高
        length_score = 1.0 / (1.0 + emd)
        
        return length_score
    
    def _compare_curvature_distributions(self, real_curvatures: np.ndarray, gen_curvatures: np.ndarray) -> float:
        """比较曲率分布"""
        real_hist, _ = np.histogram(
            real_curvatures, 
            bins=self.config['num_curvature_bins'], 
            range=(0, np.pi)
        )
        gen_hist, _ = np.histogram(
            gen_curvatures, 
            bins=self.config['num_curvature_bins'], 
            range=(0, np.pi)
        )
        
        # 归一化
        real_hist = real_hist.astype(float)
        gen_hist = gen_hist.astype(float)
        
        if np.sum(real_hist) > 0:
            real_hist = real_hist / np.sum(real_hist)
        if np.sum(gen_hist) > 0:
            gen_hist = gen_hist / np.sum(gen_hist)
        
        # 计算余弦相似度
        if np.sum(real_hist) > 0 and np.sum(gen_hist) > 0:
            cos_sim = cosine_similarity([real_hist], [gen_hist])[0][0]
            # 归一化到0-1
            curv_score = (cos_sim + 1) / 2
        else:
            curv_score = 0.5  # 默认值
        
        return curv_score
    
    def _compare_direction_distributions(self, real_directions: np.ndarray, gen_directions: np.ndarray) -> float:
        """比较方向分布"""
        # 确保直方图长度一致
        if len(real_directions) != len(gen_directions):
            min_len = min(len(real_directions), len(gen_directions))
            real_directions = real_directions[:min_len]
            gen_directions = gen_directions[:min_len]
        
        # 计算余弦相似度
        if np.sum(real_directions) > 0 and np.sum(gen_directions) > 0:
            cos_sim = cosine_similarity([real_directions], [gen_directions])[0][0]
            # 归一化到0-1
            dir_score = (cos_sim + 1) / 2
        else:
            # 如果两者都是零向量，则视为相似
            if np.sum(real_directions) == 0 and np.sum(gen_directions) == 0:
                dir_score = 1.0
            else:
                dir_score = 0.0
        
        return dir_score
    
    def _compare_spatial_features(self, real_features: Dict, gen_features: Dict) -> float:
        """比较空间特征"""
        spatial_scores = []
        
        # 1. 笔画数量相似度
        real_stroke_count = real_features.get('stroke_count', 0)
        gen_stroke_count = gen_features.get('stroke_count', 0)
        
        if real_stroke_count + gen_stroke_count > 0:
            stroke_count_score = 1.0 - abs(real_stroke_count - gen_stroke_count) / (real_stroke_count + gen_stroke_count)
            spatial_scores.append(stroke_count_score)
        
        # 2. 边界框相似度
        real_bbox = real_features.get('bbox', np.zeros(4))
        gen_bbox = gen_features.get('bbox', np.zeros(4))
        
        if np.any(real_bbox != 0) and np.any(gen_bbox != 0):
            # 计算IoU（交并比）
            x_min = max(real_bbox[0], gen_bbox[0])
            y_min = max(real_bbox[1], gen_bbox[1])
            x_max = min(real_bbox[2], gen_bbox[2])
            y_max = min(real_bbox[3], gen_bbox[3])
            
            intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
            real_area = (real_bbox[2] - real_bbox[0]) * (real_bbox[3] - real_bbox[1])
            gen_area = (gen_bbox[2] - gen_bbox[0]) * (gen_bbox[3] - gen_bbox[1])
            union = real_area + gen_area - intersection
            
            if union > 0:
                iou = intersection / union
                spatial_scores.append(iou)
        
        if spatial_scores:
            return float(np.mean(spatial_scores))
        else:
            return None
    
    def compute_consistency(self, real_sketch: np.ndarray, generated_sketch: np.ndarray) -> float:
        """
        计算真实草图与生成草图之间的结构一致性
        
        Args:
            real_sketch: 真实草图序列
            generated_sketch: 生成草图序列
            
        Returns:
            一致性分数 (0-1)，越高表示一致性越好
        """
        # 提取特征
        real_features = self._extract_stroke_features(real_sketch)
        gen_features = self._extract_stroke_features(generated_sketch)
        
        scores = []
        
        # 1. 笔画长度相似性
        if len(real_features['stroke_length_dist']) > 0 and len(gen_features['stroke_length_dist']) > 0:
            length_score = self._compare_length_distributions(
                real_features['stroke_length_dist'], 
                gen_features['stroke_length_dist']
            )
            scores.append(length_score)
        
        # 2. 曲率分布相似性
        if len(real_features['curvature_dist']) > 0 and len(gen_features['curvature_dist']) > 0:
            curvature_score = self._compare_curvature_distributions(
                real_features['curvature_dist'],
                gen_features['curvature_dist']
            )
            scores.append(curvature_score)
        
        # 3. 方向分布相似性
        direction_score = self._compare_direction_distributions(
            real_features['direction_hist'],
            gen_features['direction_hist']
        )
        scores.append(direction_score)
        
        # 4. 空间特征相似性（可选）
        spatial_score = self._compare_spatial_features(real_features, gen_features)
        if spatial_score is not None:
            scores.append(spatial_score)
        
        # 综合分数（加权平均）
        weights = self.config['feature_weights']
        if len(scores) < len(weights):
            weights = weights[:len(scores)]
        
        # 归一化权重
        weights = np.array(weights) / np.sum(weights)
        
        final_score = np.average(scores[:len(weights)], weights=weights[:len(scores)])
        
        return float(final_score)
    
    def evaluate_batch(self, real_sketches: List[np.ndarray], generated_sketches: List[np.ndarray]) -> Dict:
        """
        批量评估
        
        Args:
            real_sketches: 真实草图列表
            generated_sketches: 生成草图列表
            
        Returns:
            评估结果字典
        """
        if len(real_sketches) != len(generated_sketches):
            raise ValueError("真实草图列表和生成草图列表长度必须相同")
        
        scores = []
        feature_scores = {
            'length': [],
            'curvature': [],
            'direction': [],
            'spatial': []
        }
        
        for i, (real, gen) in enumerate(zip(real_sketches, generated_sketches)):
            # 计算综合分数
            score = self.compute_consistency(real, gen)
            scores.append(score)
            
            # 计算各特征分数
            real_features = self._extract_stroke_features(real)
            gen_features = self._extract_stroke_features(gen)
            
            # 长度分数
            if len(real_features['stroke_length_dist']) > 0 and len(gen_features['stroke_length_dist']) > 0:
                length_score = self._compare_length_distributions(
                    real_features['stroke_length_dist'], 
                    gen_features['stroke_length_dist']
                )
                feature_scores['length'].append(length_score)
            
            # 曲率分数
            if len(real_features['curvature_dist']) > 0 and len(gen_features['curvature_dist']) > 0:
                curvature_score = self._compare_curvature_distributions(
                    real_features['curvature_dist'],
                    gen_features['curvature_dist']
                )
                feature_scores['curvature'].append(curvature_score)
            
            # 方向分数
            direction_score = self._compare_direction_distributions(
                real_features['direction_hist'],
                gen_features['direction_hist']
            )
            feature_scores['direction'].append(direction_score)
            
            # 空间分数
            spatial_score = self._compare_spatial_features(real_features, gen_features)
            if spatial_score is not None:
                feature_scores['spatial'].append(spatial_score)
        
        scores = np.array(scores)
        
        return {
            'mean_consistency': float(np.mean(scores)),
            'std_consistency': float(np.std(scores)),
            'median_consistency': float(np.median(scores)),
            'min_consistency': float(np.min(scores)),
            'max_consistency': float(np.max(scores)),
            'individual_scores': scores.tolist(),
            'feature_scores': {k: (float(np.mean(v)) if v else 0.0) for k, v in feature_scores.items()},
            'histogram': self._create_histogram(scores)
        }
    
    def _create_histogram(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """创建分数直方图"""
        hist, bins = np.histogram(scores, bins=10, range=(0, 1))
        return hist.tolist(), bins.tolist()


class ReconstructionEvaluator:
    """
    重建结果评估器
    用于读取reconstruction文件夹中的原始和重建草图，并进行评估
    """
    def __init__(self, reconstruction_dir: str = "reconstruction"):
        """
        初始化评估器
        
        Args:
            reconstruction_dir: 包含重建结果的文件夹路径
        """
        self.reconstruction_dir = Path(reconstruction_dir)
        self.consistency_evaluator = StrokeStructureConsistency()
        
    def load_sketch_data(self, filepath: Path) -> np.ndarray:
        """
        加载草图数据并转换为标准格式 [N, 3]
        格式: [x, y, pen_state]，其中pen_state: 0=笔落下, 1=笔抬起
        
        注: 原始数据格式为 [N, 4]，其中:
            第0列: x坐标
            第1列: y坐标
            第2列: pen_lift (笔抬起标志，1表示笔抬起)
            第3列: pen_eos (序列结束标志)
        """
        try:
            data = np.load(filepath)
            
            # 检查数据形状
            if len(data.shape) != 2:
                print(f"警告: 数据形状异常 {data.shape}，文件: {filepath}")
                return np.array([])
            
            # 找到第一个结束点（如果有的话）
            if data.shape[1] >= 4:
                eos_indices = np.where(data[:, 3] == 1)[0]
                if len(eos_indices) > 0:
                    end_idx = eos_indices[0]
                    data = data[:end_idx, :]
            
            # 转换为标准格式 [N, 3]
            if data.shape[1] >= 3:
                # 使用pen_lift作为pen_state
                sketch_data = data[:, :3]
                # pen_lift=1表示笔抬起，所以我们直接使用
                return sketch_data
            else:
                print(f"错误: 数据列数不足 {data.shape}，文件: {filepath}")
                return np.array([])
                
        except Exception as e:
            print(f"加载文件失败 {filepath}: {e}")
            return np.array([])
    
    def find_sketch_pairs(self) -> List[Tuple[Path, Path, str, int]]:
        """
        在reconstruction文件夹中查找原始和重建文件对
        
        Returns:
            文件对列表: (original_path, recon_path, category, id)
        """
        sketch_pairs = []
        
        if not self.reconstruction_dir.exists():
            print(f"错误: 文件夹不存在 {self.reconstruction_dir}")
            return sketch_pairs
        
        # 查找所有原始文件
        original_files = list(self.reconstruction_dir.glob("original_*.npy"))
        print(f"找到 {len(original_files)} 个原始文件")
        
        for orig_file in original_files:
            # 从文件名中提取信息
            filename = orig_file.stem
            # 格式: original_{category}_id{id}
            parts = filename.split('_')
            
            if len(parts) >= 3 and parts[0] == "original":
                category = parts[1]
                # 提取id
                id_part = '_'.join(parts[2:])  # 处理id中可能包含的下划线
                
                # 构建重建文件名
                recon_filename = f"recon_{category}_{id_part}.npy"
                recon_file = self.reconstruction_dir / recon_filename
                
                if recon_file.exists():
                    sketch_pairs.append((orig_file, recon_file, category, id_part))
                else:
                    print(f"警告: 未找到对应的重建文件 {recon_filename}")
        
        print(f"找到 {len(sketch_pairs)} 个有效文件对")
        return sketch_pairs
    
    def evaluate_all_pairs(self, sketch_pairs: List[Tuple[Path, Path, str, int]]) -> Dict:
        """
        评估所有文件对
        
        Args:
            sketch_pairs: 文件对列表
            
        Returns:
            评估结果字典
        """
        results = {
            'all_scores': [],
            'category_scores': {},
            'file_pairs': []
        }
        
        for orig_path, recon_path, category, sketch_id in sketch_pairs:
            print(f"处理: {category} - {sketch_id}")
            
            # 加载数据
            orig_data = self.load_sketch_data(orig_path)
            recon_data = self.load_sketch_data(recon_path)
            
            if len(orig_data) == 0 or len(recon_data) == 0:
                print(f"警告: 数据为空，跳过 {category}_{sketch_id}")
                continue
            
            # 计算一致性分数
            consistency_score = self.consistency_evaluator.compute_consistency(orig_data, recon_data)
            
            # 存储结果
            result_entry = {
                'category': category,
                'id': sketch_id,
                'original_file': str(orig_path.name),
                'recon_file': str(recon_path.name),
                'original_points': len(orig_data),
                'recon_points': len(recon_data),
                'consistency_score': consistency_score
            }
            
            results['all_scores'].append(result_entry)
            results['file_pairs'].append((orig_path.name, recon_path.name))
            
            # 按类别统计
            if category not in results['category_scores']:
                results['category_scores'][category] = []
            results['category_scores'][category].append(consistency_score)
            
            print(f"  一致性分数: {consistency_score:.4f}, "
                  f"原始点数: {len(orig_data)}, 重建点数: {len(recon_data)}")
        
        return results
    
    def calculate_statistics(self, results: Dict) -> Dict:
        """
        计算统计信息
        
        Args:
            results: 评估结果
            
        Returns:
            统计信息字典
        """
        if not results['all_scores']:
            return {}
        
        all_scores = [entry['consistency_score'] for entry in results['all_scores']]
        
        stats = {
            'overall': {
                'mean': float(np.mean(all_scores)),
                'std': float(np.std(all_scores)),
                'median': float(np.median(all_scores)),
                'min': float(np.min(all_scores)),
                'max': float(np.max(all_scores)),
                'count': len(all_scores)
            },
            'by_category': {}
        }
        
        # 按类别统计
        for category, scores in results['category_scores'].items():
            if scores:
                stats['by_category'][category] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'median': float(np.median(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'count': len(scores)
                }
        
        return stats
    
    def visualize_results(self, results: Dict, stats: Dict, save_dir: str = None):
        """
        可视化评估结果
        
        Args:
            results: 评估结果
            stats: 统计信息
            save_dir: 保存目录
        """
        if not results['all_scores']:
            print("没有可可视化的数据")
            return
        
        all_scores = [entry['consistency_score'] for entry in results['all_scores']]
        categories = list(results['category_scores'].keys())
        
        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 总体分数分布直方图
        axes[0, 0].hist(all_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(stats['overall']['mean'], color='red', linestyle='--', 
                          label=f'Mean: {stats["overall"]["mean"]:.3f}')
        axes[0, 0].axvline(stats['overall']['median'], color='green', linestyle=':', 
                          label=f'Median: {stats["overall"]["median"]:.3f}')
        axes[0, 0].set_xlabel('Consistency Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Overall Distribution of Consistency Scores')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 按类别箱线图
        category_scores = []
        category_labels = []
        for category in categories:
            if category in results['category_scores']:
                category_scores.append(results['category_scores'][category])
                category_labels.append(category)
        
        if category_scores:
            axes[0, 1].boxplot(category_scores, labels=category_labels)
            axes[0, 1].set_ylabel('Consistency Score')
            axes[0, 1].set_title('Consistency Scores by Category')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 类别均值条形图
        if stats['by_category']:
            category_means = [stats['by_category'][cat]['mean'] for cat in categories]
            category_counts = [stats['by_category'][cat]['count'] for cat in categories]
            
            x = np.arange(len(categories))
            width = 0.6
            
            bars = axes[0, 2].bar(x, category_means, width, alpha=0.7)
            axes[0, 2].set_xlabel('Category')
            axes[0, 2].set_ylabel('Mean Consistency Score')
            axes[0, 2].set_title('Mean Consistency by Category')
            axes[0, 2].set_xticks(x)
            axes[0, 2].set_xticklabels(categories, rotation=45)
            axes[0, 2].grid(True, alpha=0.3, axis='y')
            
            # 在条形图上添加计数值
            for i, (bar, count) in enumerate(zip(bars, category_counts)):
                height = bar.get_height()
                axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'n={count}', ha='center', va='bottom', fontsize=8)
        
        # 4. 散点图：原始点数 vs 一致性分数
        orig_lengths = [entry['original_points'] for entry in results['all_scores']]
        scores = [entry['consistency_score'] for entry in results['all_scores']]
        
        scatter = axes[1, 0].scatter(orig_lengths, scores, alpha=0.6, c='blue', edgecolors='black')
        axes[1, 0].set_xlabel('Original Sketch Points Count')
        axes[1, 0].set_ylabel('Consistency Score')
        axes[1, 0].set_title('Score vs Original Points Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 添加回归线
        if len(orig_lengths) > 1:
            z = np.polyfit(orig_lengths, scores, 1)
            p = np.poly1d(z)
            axes[1, 0].plot(np.sort(orig_lengths), p(np.sort(orig_lengths)), "r--", alpha=0.8)
        
        # 5. 散点图：重建点数 vs 一致性分数
        recon_lengths = [entry['recon_points'] for entry in results['all_scores']]
        
        scatter = axes[1, 1].scatter(recon_lengths, scores, alpha=0.6, c='green', edgecolors='black')
        axes[1, 1].set_xlabel('Reconstructed Sketch Points Count')
        axes[1, 1].set_ylabel('Consistency Score')
        axes[1, 1].set_title('Score vs Reconstructed Points Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加回归线
        if len(recon_lengths) > 1:
            z = np.polyfit(recon_lengths, scores, 1)
            p = np.poly1d(z)
            axes[1, 1].plot(np.sort(recon_lengths), p(np.sort(recon_lengths)), "r--", alpha=0.8)
        
        # 6. 点数量差异 vs 一致性分数
        point_diffs = [abs(orig - recon) for orig, recon in zip(orig_lengths, recon_lengths)]
        
        scatter = axes[1, 2].scatter(point_diffs, scores, alpha=0.6, c='orange', edgecolors='black')
        axes[1, 2].set_xlabel('Absolute Points Count Difference')
        axes[1, 2].set_ylabel('Consistency Score')
        axes[1, 2].set_title('Score vs Points Count Difference')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 添加回归线
        if len(point_diffs) > 1:
            z = np.polyfit(point_diffs, scores, 1)
            p = np.poly1d(z)
            axes[1, 2].plot(np.sort(point_diffs), p(np.sort(point_diffs)), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'consistency_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_detailed_report(self, results: Dict, stats: Dict, output_dir: str = "evaluation_results"):
        """
        保存详细评估报告
        
        Args:
            results: 评估结果
            stats: 统计信息
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存详细结果到CSV
        df = pd.DataFrame(results['all_scores'])
        csv_path = os.path.join(output_dir, 'detailed_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"详细结果已保存到: {csv_path}")
        
        # 2. 保存统计信息到JSON
        json_path = os.path.join(output_dir, 'statistics.json')
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"统计信息已保存到: {json_path}")
        
        # 3. 生成并保存摘要报告
        summary_path = os.path.join(output_dir, 'summary_report.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("草图重建一致性评估报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"评估时间: {pd.Timestamp.now()}\n")
            f.write(f"评估样本总数: {len(results['all_scores'])}\n")
            f.write(f"类别数量: {len(results['category_scores'])}\n")
            f.write(f"类别: {', '.join(results['category_scores'].keys())}\n\n")
            
            f.write("总体统计:\n")
            f.write("-" * 40 + "\n")
            for key, value in stats['overall'].items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n按类别统计:\n")
            f.write("-" * 40 + "\n")
            for category, cat_stats in stats['by_category'].items():
                f.write(f"\n{category}:\n")
                for key, value in cat_stats.items():
                    f.write(f"  {key}: {value}\n")
            
            f.write("\n分数分布:\n")
            f.write("-" * 40 + "\n")
            score_ranges = {
                "优秀 (0.9-1.0)": 0,
                "良好 (0.7-0.9)": 0,
                "中等 (0.5-0.7)": 0,
                "较差 (0.3-0.5)": 0,
                "差 (0.0-0.3)": 0
            }
            
            for entry in results['all_scores']:
                score = entry['consistency_score']
                if score >= 0.9:
                    score_ranges["优秀 (0.9-1.0)"] += 1
                elif score >= 0.7:
                    score_ranges["良好 (0.7-0.9)"] += 1
                elif score >= 0.5:
                    score_ranges["中等 (0.5-0.7)"] += 1
                elif score >= 0.3:
                    score_ranges["较差 (0.3-0.5)"] += 1
                else:
                    score_ranges["差 (0.0-0.3)"] += 1
            
            for range_name, count in score_ranges.items():
                percentage = (count / len(results['all_scores'])) * 100
                f.write(f"{range_name}: {count} ({percentage:.1f}%)\n")
            
            f.write("\n前5个最佳重建:\n")
            f.write("-" * 40 + "\n")
            sorted_by_score = sorted(results['all_scores'], key=lambda x: x['consistency_score'], reverse=True)
            for i, entry in enumerate(sorted_by_score[:5]):
                f.write(f"{i+1}. {entry['category']} - {entry['id']}: {entry['consistency_score']:.4f}\n")
            
            f.write("\n前5个最差重建:\n")
            f.write("-" * 40 + "\n")
            for i, entry in enumerate(sorted_by_score[-5:]):
                f.write(f"{i+1}. {entry['category']} - {entry['id']}: {entry['consistency_score']:.4f}\n")
        
        print(f"摘要报告已保存到: {summary_path}")
        
        return {
            'csv_path': csv_path,
            'json_path': json_path,
            'summary_path': summary_path
        }
    
    def run_evaluation(self, save_results: bool = True, visualize: bool = True):
        """
        运行完整评估流程
        
        Args:
            save_results: 是否保存结果
            visualize: 是否生成可视化
            
        Returns:
            评估结果
        """
        print("=" * 60)
        print("草图重建一致性评估")
        print("=" * 60)
        
        # 1. 查找文件对
        print("\n1. 查找原始和重建文件对...")
        sketch_pairs = self.find_sketch_pairs()
        
        if not sketch_pairs:
            print("错误: 未找到有效的文件对")
            return None
        
        # 2. 评估所有文件对
        print(f"\n2. 评估 {len(sketch_pairs)} 个文件对...")
        results = self.evaluate_all_pairs(sketch_pairs)
        
        if not results['all_scores']:
            print("错误: 没有成功评估的文件对")
            return None
        
        # 3. 计算统计信息
        print("\n3. 计算统计信息...")
        stats = self.calculate_statistics(results)
        
        # 4. 打印结果摘要
        print("\n" + "=" * 60)
        print("评估结果摘要")
        print("=" * 60)
        
        overall = stats['overall']
        print(f"总体统计:")
        print(f"  平均一致性: {overall['mean']:.4f} ± {overall['std']:.4f}")
        print(f"  中位数: {overall['median']:.4f}")
        print(f"  范围: [{overall['min']:.4f}, {overall['max']:.4f}]")
        print(f"  样本数: {overall['count']}")
        
        print(f"\n按类别统计:")
        for category, cat_stats in stats['by_category'].items():
            print(f"  {category}:")
            print(f"    平均: {cat_stats['mean']:.4f} ± {cat_stats['std']:.4f}")
            print(f"    样本数: {cat_stats['count']}")
        
        # 5. 生成可视化
        if visualize:
            print("\n4. 生成可视化图表...")
            self.visualize_results(results, stats, save_dir="evaluation_results" if save_results else None)
        
        # 6. 保存结果
        if save_results:
            print("\n5. 保存评估结果...")
            saved_files = self.save_detailed_report(results, stats)
            print(f"所有结果已保存到 evaluation_results/ 文件夹")
        
        print("\n" + "=" * 60)
        print("评估完成!")
        print("=" * 60)
        
        return {
            'results': results,
            'statistics': stats
        }


def visualize_sample_pairs(results_data, reconstruction_dir="reconstruction", num_samples=5):
    """
    可视化最高分和最低分的样本对
    
    Args:
        results_data: 评估结果
        reconstruction_dir: reconstruction文件夹路径
        num_samples: 每个类别显示的样本数
    """
    if not results_data or 'all_scores' not in results_data:
        print("错误: 无结果数据")
        return
    
    # 按分数排序
    sorted_scores = sorted(results_data['all_scores'], 
                          key=lambda x: x['consistency_score'])
    
    # 获取最佳和最差样本
    worst_samples = sorted_scores[:min(num_samples, len(sorted_scores))]
    best_samples = sorted_scores[-min(num_samples, len(sorted_scores)):]
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    # 可视化最差样本
    print("\n最差重建样本 (最低分):")
    for i, sample in enumerate(worst_samples):
        if i >= num_samples:
            break
            
        # 加载数据
        recon_dir = Path(reconstruction_dir)
        orig_path = recon_dir / sample['original_file']
        recon_path = recon_dir / sample['recon_file']
        
        evaluator = ReconstructionEvaluator()
        orig_data = evaluator.load_sketch_data(orig_path)
        recon_data = evaluator.load_sketch_data(recon_path)
        
        # 绘制原始草图
        axes[0, i].plot(orig_data[:, 0], -orig_data[:, 1], 'b-', linewidth=2, label='Original')
        axes[0, i].set_title(f"Original: {sample['category']}\nScore: {sample['consistency_score']:.3f}")
        axes[0, i].axis('equal')
        axes[0, i].grid(True, alpha=0.3)
        
        # 绘制重建草图
        axes[1, i].plot(recon_data[:, 0], -recon_data[:, 1], 'r-', linewidth=2, label='Reconstructed')
        axes[1, i].set_title(f"Reconstructed")
        axes[1, i].axis('equal')
        axes[1, i].grid(True, alpha=0.3)
        
        print(f"  {sample['category']} - {sample['id']}: {sample['consistency_score']:.4f}")
    
    plt.suptitle(f"Worst {num_samples} Reconstructions (Lowest Scores)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig("evaluation_results/worst_reconstructions.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 可视化最佳样本
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    print("\n最佳重建样本 (最高分):")
    for i, sample in enumerate(reversed(best_samples)):
        if i >= num_samples:
            break
            
        # 加载数据
        recon_dir = Path(reconstruction_dir)
        orig_path = recon_dir / sample['original_file']
        recon_path = recon_dir / sample['recon_file']
        
        evaluator = ReconstructionEvaluator()
        orig_data = evaluator.load_sketch_data(orig_path)
        recon_data = evaluator.load_sketch_data(recon_path)
        
        # 绘制原始草图
        axes[0, i].plot(orig_data[:, 0], -orig_data[:, 1], 'b-', linewidth=2, label='Original')
        axes[0, i].set_title(f"Original: {sample['category']}\nScore: {sample['consistency_score']:.3f}")
        axes[0, i].axis('equal')
        axes[0, i].grid(True, alpha=0.3)
        
        # 绘制重建草图
        axes[1, i].plot(recon_data[:, 0], -recon_data[:, 1], 'g-', linewidth=2, label='Reconstructed')
        axes[1, i].set_title(f"Reconstructed")
        axes[1, i].axis('equal')
        axes[1, i].grid(True, alpha=0.3)
        
        print(f"  {sample['category']} - {sample['id']}: {sample['consistency_score']:.4f}")
    
    plt.suptitle(f"Best {num_samples} Reconstructions (Highest Scores)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig("evaluation_results/best_reconstructions.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # 运行评估
    evaluator = ReconstructionEvaluator(reconstruction_dir="reconstruction")
    
    # 执行评估
    evaluation_results = evaluator.run_evaluation(save_results=True, visualize=True)
    
    if evaluation_results:
        # 可视化样本
        #visualize_sample_pairs(evaluation_results['results'], num_samples=5)
        
        # 打印详细统计
        print("\n详细统计信息:")
        print("-" * 40)
        
        all_scores = [entry['consistency_score'] for entry in evaluation_results['results']['all_scores']]
        
        # 计算百分位数
        #percentiles = [25, 50, 75, 90, 95]
        #percentile_values = np.percentile(all_scores, percentiles)
        
        #for p, val in zip(percentiles, percentile_values):
        #    print(f"  {p}th 百分位数: {val:.4f}")
        
        # 分数分布
        print("\n分数分布:")
        score_ranges = [(0.9, 1.0), (0.7, 0.9), (0.5, 0.7), (0.3, 0.5), (0.0, 0.3)]
        range_names = ["优秀", "良好", "中等", "较差", "差"]
        
        for (low, high), name in zip(score_ranges, range_names):
            count = sum(1 for score in all_scores if low <= score < high)
            percentage = (count / len(all_scores)) * 100
            print(f"  {name} ({low:.1f}-{high:.1f}): {count} 个样本 ({percentage:.1f}%)")
        
        # 按类别分析
        print("\n按类别分析:")
        for category, scores in evaluation_results['results']['category_scores'].items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"  {category}: {len(scores)} 个样本, 平均分: {mean_score:.4f} ± {std_score:.4f}")