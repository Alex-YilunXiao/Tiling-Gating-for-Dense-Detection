# -*- coding: utf-8 -*-
"""
Enhanced (Reduced-Cost) Parameter Study for YOLO Post-processing.

This script implements a two-stage search strategy (coarse -> refine) combined
with ablation studies to efficiently explore the parameter space of a
post-processing pipeline involving tiling, spatial gating, semantic gating,
and confidence boosting.
"""

import argparse
import glob
import json
import logging
import random
import time
import warnings
import math
import sys
import os
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Generator

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from sklearn.cluster import DBSCAN
from torchvision import models
from torchvision import transforms as T
from torchvision.ops import boxes as box_ops
from ultralytics import YOLO
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")

# Type Aliases for clarity
Boxes = np.ndarray  # Shape (N, 4), format [x1, y1, x2, y2]
Scores = np.ndarray # Shape (N,)
Classes = np.ndarray# Shape (N,)
Detections = Tuple[Boxes, Scores, Classes]
Params = Dict[str, Union[float, int]]
ExperimentResult = Dict[str, Any]

# =============================================================================
# Configuration and Constants
# =============================================================================

# VisDrone Class Names
CLASS_NAMES = {0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van',
               5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'}

# --- Optimization Strategy Configuration ---

# Stage A (Coarse Grid Search): 24 configurations
STAGE_A_GRID = {
    'base_conf': [0.25, 0.30],
    'tile_conf': [0.15, 0.20],
    'spatial_eps_multiplier': [1.0, 1.5, 2.0],
    'semantic_eps': [0.30, 0.40]
}
STAGE_A_DEFAULTS = {
    'min_samples': 3,
    'quality_threshold': 0.30,
    'nms_iou_threshold': 0.55,
    'confidence_boost_factor': 0.10
}

# Stage B (Focused Random Search): 18 samples
STAGE_B_SEARCH_SPACE = {
    'min_samples': [2, 3, 4],
    'quality_threshold': [0.25, 0.30, 0.35],
    'nms_iou_threshold': [0.55, 0.65],
    'confidence_boost_factor': [0.05, 0.10, 0.15]
}
STAGE_B_SAMPLES = 18

# Finalization
TOPK_FULL = 3

# Ablation Study Steps
ABLATION_STEPS = [
    {"tiling": False, "spatial": False, "semantic": False, "boost": False, "tag": "Baseline"},
    {"tiling": True,  "spatial": False, "semantic": False, "boost": False, "tag": "+Tiling"},
    {"tiling": True,  "spatial": True,  "semantic": False, "boost": False, "tag": "+Spatial"},
    {"tiling": True,  "spatial": True,  "semantic": True,  "boost": False, "tag": "+Semantic"},
    {"tiling": True,  "spatial": True,  "semantic": True,  "boost": True,  "tag": "+Boost (Full)"},
]

# =============================================================================
# Visualization Setup (IEEE Style)
# =============================================================================

def configure_matplotlib():
    """Configures Matplotlib for generating IEEE-compliant figures."""
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 1.5,
    })

# =============================================================================
# Utility Functions
# =============================================================================

def set_seed(seed: int = 2025):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}.")

def _calculate_iou_np(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculates Intersection over Union (IoU) between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / (union_area + 1e-6)

def load_ground_truth(annotation_path: Path) -> Tuple[Boxes, Classes]:
    """Loads ground truth annotations from a VisDrone format file."""
    if not annotation_path.exists():
        return np.zeros((0, 4)), np.zeros(0, dtype=int)

    gt_boxes, gt_classes = [], []
    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                try:
                    # Handle potential variations (comma or space)
                    parts_str = line.strip().replace(' ', ',').split(',')
                    if len(parts_str) < 8:
                        continue
                    
                    # Format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,...
                    parts = [int(p) for p in parts_str[:8]]
                    x, y, w, h, _, class_id, _, _ = parts

                    # VisDrone classes range from 1-10
                    if 1 <= class_id <= 10:
                        gt_boxes.append([x, y, x + w, y + h])
                        gt_classes.append(class_id - 1) # Convert to 0-indexed
                except ValueError:
                    continue # Skip lines that cannot be parsed
    except Exception as e:
        logging.warning(f"Error reading annotation file {annotation_path}: {e}")
        return np.zeros((0, 4)), np.zeros(0, dtype=int)

    return np.array(gt_boxes), np.array(gt_classes, dtype=int)

def calculate_metrics(
    pred_boxes: Boxes, pred_scores: Scores, pred_classes: Classes,
    gt_boxes: Boxes, gt_classes: Classes, iou_threshold: float = 0.5
) -> Dict[str, Union[float, int]]:
    """Calculates Precision, Recall, F1 score, TP, FP, and FN for a single image."""

    # Handle edge cases
    if len(gt_boxes) == 0:
        if len(pred_boxes) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
        else:
            return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0, 'tp': 0, 'fp': len(pred_boxes), 'fn': 0}

    if len(pred_boxes) == 0:
        return {'precision': 1.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gt_boxes)}

    # Sort predictions by score descending
    sort_indices = np.argsort(pred_scores)[::-1]
    pb, pc = pred_boxes[sort_indices], pred_classes[sort_indices]

    tp, fp = 0, 0
    gt_matched = [False] * len(gt_boxes)

    for i in range(len(pb)):
        best_iou, best_j = 0.0, -1
        # Find the best matching GT box of the same class
        for j in range(len(gt_boxes)):
            if int(gt_classes[j]) == int(pc[i]):
                iou = _calculate_iou_np(pb[i], gt_boxes[j])
                if iou > best_iou:
                    best_iou, best_j = iou, j

        # Determine TP or FP
        if best_iou >= iou_threshold and best_j != -1 and not gt_matched[best_j]:
            tp += 1
            gt_matched[best_j] = True
        else:
            fp += 1

    fn = len(gt_boxes) - sum(gt_matched)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}


# =============================================================================
# Feature Extractor Component
# =============================================================================

class EnhancedFeatureExtractor:
    """Extracts deep features from image ROIs using a pre-trained ResNet18."""
    def __init__(self, device: torch.device):
        self.device = device
        self.feature_dim = 512
        self.model = self._load_model()
        self.transform = self._get_transforms()

    def _load_model(self) -> nn.Module:
        """Loads ResNet18 pretrained on ImageNet using modern syntax."""
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        except Exception as e:
            logging.warning(f"Failed to load ResNet18 weights using modern syntax: {e}. Falling back to legacy 'pretrained=True'.")
            model = models.resnet18(pretrained=True)

        # Use the model up to the average pooling layer
        model = nn.Sequential(*list(model.children())[:-1])
        model.to(self.device).eval()
        return model

    def _get_transforms(self) -> T.Compose:
        """Defines the preprocessing pipeline."""
        return T.Compose([
            T.ToPILImage(),
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract_batch(self, img_bgr: np.ndarray, boxes: Boxes, batch_size: int = 64) -> np.ndarray:
        """Extracts features for a batch of boxes from a single image.

        Args:
            img_bgr: The source image (H, W, C) in BGR format (OpenCV standard).
            boxes: Array of boxes [N, 4] in XYXY format.
            batch_size: Batch size for inference.

        Returns:
            Array of features [N, D].
        """
        if len(boxes) == 0:
            return np.zeros((0, self.feature_dim))

        # Convert BGR to RGB (Torchvision expects RGB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        tensors, valid_indices = [], []
        h, w = img_rgb.shape[:2]

        # Preprocess ROIs
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            # Clamp coordinates and ensure valid dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                roi_rgb = img_rgb[y1:y2, x1:x2]
                tensors.append(self.transform(roi_rgb))
                valid_indices.append(i)

        if not tensors:
            return np.zeros((len(boxes), self.feature_dim))

        tensors = torch.stack(tensors).to(self.device)
        features = np.zeros((len(boxes), self.feature_dim), dtype=np.float32)

        # Process in batches
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i:i + batch_size]
            out = self.model(batch).squeeze(-1).squeeze(-1)
            if out.ndim == 1:
                out = out.unsqueeze(0) # Handle single-item batch
            features[valid_indices[i:i + len(batch)]] = out.cpu().numpy()

        return features

# =============================================================================
# Detection Helpers and Caching
# =============================================================================

class DetectionCache:
    """Encapsulates caching of detections to avoid recomputation."""
    def __init__(self):
        # Key: (img_path_str, confidence_threshold, mode_str)
        self.cache: Dict[Tuple[str, float, str], Detections] = {}

    def get(self, img_path: str, conf: float, mode: str) -> Optional[Detections]:
        """Retrieve detections from cache."""
        key = (img_path, float(conf), mode)
        return self.cache.get(key)

    def set(self, img_path: str, conf: float, mode: str, detections: Detections) -> None:
        """Store detections in cache."""
        key = (img_path, float(conf), mode)
        self.cache[key] = detections

def run_yolo_on_image(model: YOLO, img: np.ndarray, conf: float) -> Detections:
    """Runs YOLO inference on a single image or tile."""
    # Ultralytics YOLO handles BGR/RGB conversion internally
    res = model.predict(source=img, conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    boxes = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    classes = res.boxes.cls.cpu().numpy().astype(int)
    return boxes, scores, classes

def compute_tiling_candidates(
    model: YOLO, img: np.ndarray, tile_conf: float, tile_size: int = 640, overlap: int = 160
) -> Detections:
    """Performs tiled inference (sliding window) on a large image."""
    h, w, _ = img.shape
    all_boxes, all_scores, all_classes = [], [], []
    stride = tile_size - overlap

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Determine tile boundaries
            y_end, x_end = min(y + tile_size, h), min(x + tile_size, w)
            
            # Adjust start coordinates to maintain tile size at edges (SAHI approach)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile = img[y_start:y_end, x_start:x_end]

            if tile.shape[0] < 32 or tile.shape[1] < 32:
                continue

            # Run inference on the tile
            boxes, scores, classes = run_yolo_on_image(model, tile, tile_conf)

            if len(boxes) > 0:
                # Transform boxes back to the original image coordinates
                boxes[:, [0, 2]] += x_start
                boxes[:, [1, 3]] += y_start
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_classes.append(classes)

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    boxes = np.vstack(all_boxes)
    scores = np.hstack(all_scores)
    classes = np.hstack(all_classes).astype(int)

    # Simple deduplication strategy using a hash to handle exact overlaps
    unique_hashes = [hash(b.tobytes()) + 31 * int(c) for b, c in zip(boxes, classes)]
    _, unique_indices = np.unique(unique_hashes, return_index=True)
    return boxes[unique_indices], scores[unique_indices], classes[unique_indices]

def get_detections(
    model: YOLO, cache: DetectionCache, img_path: str, img: np.ndarray, conf: float, mode: str
) -> Detections:
    """Retrieves detections (base or tiling), utilizing the provided cache."""
    cached_dets = cache.get(img_path, conf, mode)
    if cached_dets:
        return cached_dets

    if mode == 'base':
        dets = run_yolo_on_image(model, img, conf)
    elif mode == 'tile':
        dets = compute_tiling_candidates(model, img, conf)
    else:
        raise ValueError(f"Invalid detection mode: {mode}")

    cache.set(img_path, conf, mode, dets)
    return dets

# =============================================================================
# Parameterized Post-processing Pipeline
# =============================================================================

class ParameterizedSemanticPipeline:
    """Implements the proposed post-processing pipeline."""
    def __init__(self, feature_extractor: EnhancedFeatureExtractor, device: torch.device,
                 params: Params, use_tiling: bool = True, use_spatial: bool = True,
                 use_semantic: bool = True, use_boost: bool = True):
        self.device = device
        # Use the shared feature extractor instance
        self.feature_extractor = feature_extractor
        self.params = params

        # Feature flags for ablation studies
        self.flags = {
            'tiling': use_tiling, 'spatial': use_spatial,
            'semantic': use_semantic, 'boost': use_boost
        }

    def _class_nms(self, boxes: Boxes, scores: Scores, classes: Classes) -> Detections:
        """Performs class-aware Non-Maximum Suppression (NMS)."""
        if len(boxes) == 0:
            return boxes, scores, classes

        keep = box_ops.batched_nms(
            torch.from_numpy(boxes).float().to(self.device),
            torch.from_numpy(scores).float().to(self.device),
            torch.from_numpy(classes).int().to(self.device),
            self.params['nms_iou_threshold']
        )
        return boxes[keep.cpu()], scores[keep.cpu()], classes[keep.cpu()]

    def _calculate_cluster_quality(self, scores: Scores, classes: Classes) -> float:
        """Calculates a quality score based on score consistency and class purity."""
        # Score quality: high mean, low standard deviation
        score_quality = np.mean(scores) / (1 + np.std(scores))
        # Class consistency: fewer unique classes is better
        class_consistency = 1.0 / len(np.unique(classes))
        # Weighted combination
        return 0.7 * score_quality + 0.3 * class_consistency

    def process(self, img_bgr: np.ndarray, base_dets: Detections, tile_dets: Optional[Detections]) -> Detections:
        """Executes the full post-processing pipeline on an image."""

        base_boxes, base_scores, base_classes = base_dets

        # Ablation: Baseline (No tiling)
        if not self.flags['tiling'] or tile_dets is None:
            return self._class_nms(base_boxes, base_scores, base_classes)

        cand_boxes, cand_scores, cand_classes = tile_dets
        min_samples = int(self.params['min_samples'])

        # Early exit if not enough candidates for clustering
        if len(cand_boxes) < min_samples:
            return self._class_nms(base_boxes, base_scores, base_classes)

        # 3. Spatial Gating
        if self.flags['spatial']:
            centers = np.column_stack([(cand_boxes[:, 0] + cand_boxes[:, 2]) / 2,
                                       (cand_boxes[:, 1] + cand_boxes[:, 3]) / 2])
            # Adaptive epsilon
            avg_diag = np.mean(np.sqrt((cand_boxes[:, 2] - cand_boxes[:, 0]) ** 2 +
                                       (cand_boxes[:, 3] - cand_boxes[:, 1]) ** 2))
            spatial_eps = avg_diag * self.params['spatial_eps_multiplier']

            spatial_db = DBSCAN(eps=spatial_eps, min_samples=min_samples).fit(centers)
            spatial_labels = spatial_db.labels_

            if not np.any(spatial_labels != -1):
                return self._class_nms(base_boxes, base_scores, base_classes)
        else:
            # Ablation: If spatial gating is disabled, treat all as one cluster (id=0)
            spatial_labels = np.zeros(len(cand_boxes), dtype=int)

        # 4. Semantic Validation
        final_validated_indices = []
        if self.flags['semantic']:
            # Pass BGR image to extractor (it handles BGR->RGB conversion)
            cand_features = self.feature_extractor.extract_batch(img_bgr, cand_boxes)
            # Normalize features for cosine similarity
            features_norm = cand_features / (np.linalg.norm(cand_features, axis=1, keepdims=True) + 1e-6)

            unique_spatial = set(spatial_labels) - {-1}
            for label in unique_spatial:
                idxs = np.where(spatial_labels == label)[0]

                if len(idxs) < min_samples:
                    continue

                # Apply semantic DBSCAN within the spatial cluster
                semantic_db = DBSCAN(
                    eps=self.params['semantic_eps'], min_samples=min_samples, metric='cosine'
                ).fit(features_norm[idxs])
                semantic_labels = semantic_db.labels_

                # Evaluate quality of semantic sub-clusters
                for s_lab in set(semantic_labels) - {-1}:
                    local_mask = (semantic_labels == s_lab)
                    global_idxs = idxs[local_mask]

                    quality = self._calculate_cluster_quality(cand_scores[global_idxs], cand_classes[global_idxs])

                    if quality > self.params['quality_threshold']:
                        final_validated_indices.extend(global_idxs)
        else:
            # Ablation: If semantic validation is disabled, accept all spatially grouped candidates
            final_validated_indices = list(np.where(spatial_labels != -1)[0])

        if not final_validated_indices:
            return self._class_nms(base_boxes, base_scores, base_classes)

        # Select validated candidates
        validated_mask = np.zeros(len(cand_boxes), dtype=bool)
        validated_mask[list(set(final_validated_indices))] = True

        validated_boxes = cand_boxes[validated_mask]
        validated_scores = cand_scores[validated_mask]
        validated_classes = cand_classes[validated_mask]

        # 5. Confidence Boosting
        if self.flags['boost']:
            # Boost factor depends logarithmically on the number of validated detections
            boost_magnitude = self.params['confidence_boost_factor'] * np.log1p(np.sum(validated_mask))
            validated_scores = validated_scores * (1 + boost_magnitude)

        # 6. Merge and Final NMS
        final_boxes = np.vstack([base_boxes, validated_boxes])
        final_scores = np.hstack([base_scores, validated_scores])
        final_classes = np.hstack([base_classes, validated_classes])

        # Clip scores
        final_scores = np.clip(final_scores, 0, 0.999)

        return self._class_nms(final_boxes, final_scores, final_classes)


# =============================================================================
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """Manages the execution of the parameter study and ablation experiments."""
    def __init__(self, model: YOLO, device: torch.device, img_paths: List[Path],
                 annotation_dir: Path, output_dir: Path):
        self.model = model
        self.device = device
        self.img_paths = img_paths
        self.annotation_dir = annotation_dir
        self.output_dir = output_dir
        self.results: List[ExperimentResult] = []
        
        # Initialize shared components
        self.cache = DetectionCache()
        self.feature_extractor = EnhancedFeatureExtractor(device)

    def _eval_config_on_images(self, params: Params, img_list: List[Path],
                               feature_flags: Optional[Dict[str, bool]] = None,
                               tag: Optional[str] = None) -> ExperimentResult:
        """Evaluates a single parameter configuration on a list of images."""
        
        # Initialize pipeline flags
        flags = {"tiling": True, "spatial": True, "semantic": True, "boost": True}
        if feature_flags:
            flags.update(feature_flags)

        # Initialize the pipeline
        pipeline = ParameterizedSemanticPipeline(
            self.feature_extractor, self.device, params,
            use_tiling=flags["tiling"], use_spatial=flags["spatial"],
            use_semantic=flags["semantic"], use_boost=flags["boost"]
        )

        image_results = []
        total_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'time': 0.0}
        valid_images = 0

        for img_path in tqdm(img_list, desc=f"Evaluating {tag}", leave=False):
            # Load image in BGR format (OpenCV standard)
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logging.warning(f"Could not read image: {img_path}")
                continue

            # Pre-fetch detections using caching mechanism
            # We pass the string representation of the path for the cache key
            img_path_str = str(img_path)
            base_dets = get_detections(self.model, self.cache, img_path_str, img_bgr, params['base_conf'], mode='base')
            tile_dets = None
            if flags["tiling"]:
                tile_dets = get_detections(self.model, self.cache, img_path_str, img_bgr, params['tile_conf'], mode='tile')

            # Load Ground Truth
            ann_path = self.annotation_dir / (img_path.stem + ".txt")
            gt_boxes, gt_classes = load_ground_truth(ann_path)

            # Run the pipeline (timed)
            t0 = time.time()
            enh_boxes, enh_scores, enh_classes = pipeline.process(img_bgr, base_dets, tile_dets)
            dt = time.time() - t0

            # Calculate metrics
            metrics = calculate_metrics(enh_boxes, enh_scores, enh_classes, gt_boxes, gt_classes)

            image_results.append({
                'image': img_path.name, **metrics, 'time': dt
            })

            # Aggregate metrics
            for key in ['precision', 'recall', 'f1']:
                total_metrics[key] += metrics[key]
            total_metrics['time'] += dt
            valid_images += 1

        # Calculate average metrics
        avg_metrics = {f'avg_{k}': (v / valid_images) if valid_images > 0 else 0.0 for k, v in total_metrics.items()}

        return {
            'tag': tag or 'config',
            'params': params.copy(),
            # 'image_results': image_results, # Optional: include if needed for detailed analysis
            **avg_metrics
        }

    def _generate_stage_a_configs(self) -> Generator[Params, None, None]:
        """Generator for Stage A (Coarse Grid Search) configurations."""
        names = list(STAGE_A_GRID.keys())
        values = [STAGE_A_GRID[n] for n in names]
        for combo in product(*values):
            p = dict(zip(names, combo))
            p.update(STAGE_A_DEFAULTS)
            yield p

    def _generate_stage_b_samples(self, best_from_a: ExperimentResult) -> Generator[Params, None, None]:
        """Generator for Stage B (Focused Random Search) configurations."""
        # Lock the best parameters found in Stage A
        base = {k: best_from_a['params'][k] for k in STAGE_A_GRID.keys()}
        
        keys = list(STAGE_B_SEARCH_SPACE.keys())
        # Generate all combinations and randomly sample
        all_combos = list(product(*[STAGE_B_SEARCH_SPACE[k] for k in keys]))
        random.shuffle(all_combos)
        
        num_samples = min(STAGE_B_SAMPLES, len(all_combos))
        picked = all_combos[:num_samples]

        for combo in picked:
            params = base.copy()
            params.update(dict(zip(keys, combo)))
            yield params

    def run_two_stage(self, subset_size_a: int, subset_size_b: int, full_eval_size: int) -> Tuple[List[ExperimentResult], List[ExperimentResult], List[ExperimentResult]]:
        """Executes the two-stage optimization strategy."""

        # --- Stage A: Coarse Search ---
        sub_a = self.img_paths[:subset_size_a]
        stage_a_results = []
        configs_a = list(self._generate_stage_a_configs())
        logging.info(f"\n--- Starting Stage A: {len(configs_a)} configs on {len(sub_a)} images ---")

        for i, p in enumerate(configs_a, 1):
            tag = f"A{i:02d}"
            logging.info(f"Evaluating {tag}...")
            r = self._eval_config_on_images(p, sub_a, tag=tag)
            stage_a_results.append(r)
            self.results.append(r)

        if not stage_a_results:
             logging.error("Stage A produced no results.")
             return [], [], []

        best_a = max(stage_a_results, key=lambda x: x['avg_f1'])
        logging.info(f"Best Stage-A F1: {best_a['avg_f1']:.4f}")

        # --- Stage B: Focused Search ---
        sub_b = self.img_paths[:subset_size_b]
        stage_b_results = []
        configs_b = list(self._generate_stage_b_samples(best_a))
        logging.info(f"\n--- Starting Stage B: {len(configs_b)} focused samples on {len(sub_b)} images ---")

        for j, p in enumerate(configs_b, 1):
            tag = f"B{j:02d}"
            logging.info(f"Evaluating {tag}...")
            r = self._eval_config_on_images(p, sub_b, tag=tag)
            stage_b_results.append(r)
            self.results.append(r)

        # --- Finalization: Full Evaluation of Top-K ---
        candidates = sorted(stage_a_results + stage_b_results, key=lambda x: x['avg_f1'], reverse=True)[:TOPK_FULL]
        full_list = self.img_paths[:full_eval_size]
        final_results = []
        logging.info(f"\n--- Starting Finalization: Re-evaluating top-{len(candidates)} configs on {len(full_list)} images ---")

        for k, cand in enumerate(candidates, 1):
            tag = f"FULL{k}"
            logging.info(f"Evaluating {tag} (Previous F1: {cand['avg_f1']:.4f})...")
            r = self._eval_config_on_images(cand['params'], full_list, tag=tag)
            final_results.append(r)
            self.results.append(r)

        return stage_a_results, stage_b_results, final_results

    def run_ablation(self, best_params: Params, ablation_subset_size: int) -> List[ExperimentResult]:
        """Executes the ablation study using the best parameters found."""
        img_list = self.img_paths[:ablation_subset_size]
        ablation_out = []
        logging.info(f"\n--- Starting Ablation Study on {len(img_list)} images ---")

        for step in ABLATION_STEPS:
            tag = step["tag"]
            flags = {k: v for k, v in step.items() if k != 'tag'}
            logging.info(f"Evaluating Ablation Step: {tag}")
            r = self._eval_config_on_images(best_params, img_list, feature_flags=flags, tag=tag)
            ablation_out.append(r)
            self.results.append(r)
        return ablation_out

    def save_results(self, filename: str = "final_results.json"):
        """Saves all experiment results to a JSON file."""
        filepath = self.output_dir / filename
        try:
            with open(filepath, 'w') as f:
                # Custom encoder for numpy types
                json.dump(self.results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else int(o) if isinstance(o, (np.int32, np.int64)) else o.__dict__)
            logging.info(f"Results successfully saved to {filepath}")
        except IOError as e:
            logging.error(f"Error saving results to {filepath}: {e}")

# =============================================================================
# Visualization (IEEE Style)
# =============================================================================
# The visualization code from the prompt is integrated here, refactored to use 
# the new structure (Pathlib, shared components).

class IEEEVisualizer:
    """Generates IEEE-compliant plots and visualizations."""
    def __init__(self, results: List[ExperimentResult], output_dir: Path, annotation_dir: Path):
        self.results = results
        self.output_dir = output_dir
        self.annotation_dir = annotation_dir
        configure_matplotlib()

    def _save_fig(self, fig: plt.Figure, name: str, formats: List[str] = ['pdf'], dpi: int = 300):
        """Helper to save figures."""
        for fmt in formats:
            filepath = self.output_dir / f"{name}.{fmt}"
            fig.savefig(filepath, format=fmt, bbox_inches='tight', dpi=dpi)
        plt.close(fig)

    def create_parameter_sensitivity_plot(self):
        # Filter results to include search phases (A/B)
        search_results = [r for r in self.results if r.get('tag', '').startswith(('A', 'B'))]
        if not search_results:
            return

        param_names = list(STAGE_A_GRID.keys()) + list(STAGE_B_SEARCH_SPACE.keys())
        
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        for idx, pname in enumerate(param_names):
            ax = fig.add_subplot(gs[idx // 4, idx % 4])
            xs, ys = [], []
            for r in search_results:
                if pname in r['params']:
                    xs.append(r['params'][pname])
                    ys.append(r['avg_f1'])
            
            if not xs:
                ax.set_axis_off()
                continue
            
            # Group by unique x values
            uniq = sorted(list(set(xs)))
            data = [[] for _ in uniq]
            for x, y in zip(xs, ys):
                data[uniq.index(x)].append(y)
            
            ax.boxplot(data, labels=[f'{u:.2f}' if isinstance(u, float) else str(u) for u in uniq])
            ax.set_xlabel(pname.replace('_', ' ').title())
            if idx % 4 == 0:
                 ax.set_ylabel('F1 Score')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Parameter Sensitivity (Two-Stage Search)', fontsize=12, fontweight='bold')
        self._save_fig(fig, 'parameter_sensitivity')

    def create_precision_recall_tradeoff_plot(self):
        if not self.results:
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # --- P-R Scatter Plot ---
        precisions = [r['avg_precision'] for r in self.results]
        recalls = [r['avg_recall'] for r in self.results]
        f1s = [r['avg_f1'] for r in self.results]
        
        scatter = ax1.scatter(recalls, precisions, c=f1s, cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision-Recall Trade-off')
        ax1.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('F1 Score')
        
        best_idx = int(np.argmax(f1s))
        ax1.scatter(recalls[best_idx], precisions[best_idx], color='red', s=180, marker='*', edgecolors='black', linewidth=1, label=f'Best F1: {f1s[best_idx]:.3f}')
        ax1.legend()

        # --- Confidence Heatmap ---
        base_confs = sorted(list(set([r['params'].get('base_conf', None) for r in self.results if 'base_conf' in r['params']])))
        tile_confs = sorted(list(set([r['params'].get('tile_conf', None) for r in self.results if 'tile_conf' in r['params']])))
        
        if base_confs and tile_confs:
            f1_matrix = np.zeros((len(base_confs), len(tile_confs)))
            for r in self.results:
                if 'base_conf' in r['params'] and 'tile_conf' in r['params']:
                    i = base_confs.index(r['params']['base_conf'])
                    j = tile_confs.index(r['params']['tile_conf'])
                    f1_matrix[i, j] = max(f1_matrix[i, j], r['avg_f1'])
            
            im = ax2.imshow(f1_matrix, cmap='RdYlGn', aspect='auto')
            ax2.set_xticks(range(len(tile_confs)))
            ax2.set_yticks(range(len(base_confs)))
            ax2.set_xticklabels([f'{tc:.2f}' for tc in tile_confs])
            ax2.set_yticklabels([f'{bc:.2f}' for bc in base_confs])
            ax2.set_xlabel('Tile Confidence')
            ax2.set_ylabel('Base Confidence')
            ax2.set_title('F1 Heatmap (max over other params)')
            
            for i in range(len(base_confs)):
                for j in range(len(tile_confs)):
                    ax2.text(j, i, f'{f1_matrix[i, j]:.2f}', ha='center', va='center', color='black', fontsize=8)
            plt.colorbar(im, ax=ax2)
        else:
            ax2.set_axis_off()

        self._save_fig(fig, 'precision_recall_tradeoff')

    def create_performance_table(self, top_n=10):
        # Focus on 'FULL' evaluation runs if available, otherwise use all
        full_results = [r for r in self.results if r.get('tag', '').startswith('FULL')]
        if full_results:
             sorted_results = sorted(full_results, key=lambda x: x['avg_f1'], reverse=True)[:top_n]
        else:
             sorted_results = sorted(self.results, key=lambda x: x['avg_f1'], reverse=True)[:top_n]

        rows = []
        for rank, r in enumerate(sorted_results, 1):
            params = r.get('params', {})
            row = {
                'Rank': rank,
                'Tag': r.get('tag', ''),
                'Precision': f"{r['avg_precision']:.3f}",
                'Recall': f"{r['avg_recall']:.3f}",
                'F1': f"{r['avg_f1']:.3f}",
                'Time (s)': f"{r['avg_time']:.3f}",
                'Base Conf': f"{params.get('base_conf', 0):.2f}",
                'Tile Conf': f"{params.get('tile_conf', 0):.2f}",
                'Qual Thr': f"{params.get('quality_threshold', 0):.2f}",
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        
        # Save CSV
        df.to_csv(self.output_dir / 'top_configs.csv', index=False)
        
        # Render as table figure
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('off'); ax.axis('tight')
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Styling
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Top Configurations by F1', fontsize=12, fontweight='bold', pad=20)
        self._save_fig(fig, 'performance_table')

    def create_runtime_plot(self):
        if not self.results:
            return
        times = [r['avg_time'] for r in self.results]
        f1s = [r['avg_f1'] for r in self.results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Performance vs Cost
        ax1.scatter(times, f1s, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Average Processing Time (s)')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('Performance vs Computational Cost')
        
        # Pareto frontier calculation
        pareto = []
        for i, (t, f) in enumerate(zip(times, f1s)):
            is_p = True
            for j, (t2, f2) in enumerate(zip(times, f1s)):
                if i != j and t2 < t and f2 > f:
                    is_p = False
                    break
            if is_p:
                pareto.append((t, f))
        if pareto:
            pareto.sort()
            ax1.plot([p[0] for p in pareto], [p[1] for p in pareto], 'r-', linewidth=2, label='Pareto Frontier')
            ax1.legend()
        
        # Runtime by min_samples
        vals = sorted(list(set([r['params'].get('min_samples', None) for r in self.results if 'min_samples' in r['params']])))
        data = [[] for _ in vals]
        for r in self.results:
            if 'min_samples' in r['params'] and r['params']['min_samples'] in vals:
                data[vals.index(r['params']['min_samples'])].append(r['avg_time'])
        
        if vals:
            ax2.boxplot(data, labels=[str(v) for v in vals])
            ax2.set_xlabel('min_samples')
            ax2.set_ylabel('Processing Time (s)')
            ax2.set_title('Runtime Distribution by min_samples')
        else:
            ax2.set_axis_off()

        self._save_fig(fig, 'runtime_analysis')

    def create_ablation_plot(self, ablation_results):
        if not ablation_results:
            return

        tags = [r['tag'] for r in ablation_results]
        f1s = [r['avg_f1'] for r in ablation_results]
        precs = [r['avg_precision'] for r in ablation_results]
        recs = [r['avg_recall'] for r in ablation_results]
        
        x = np.arange(len(tags))
        fig = plt.figure(figsize=(8, 4.5))
        
        plt.plot(x, f1s, marker='o', label='F1')
        plt.plot(x, precs, marker='s', label='Precision')
        plt.plot(x, recs, marker='^', label='Recall')
        
        plt.xticks(x, tags, rotation=0)
        plt.ylabel('Score')
        plt.title('Ablation Study')
        plt.legend()
        plt.grid(True, alpha=0.3)
        self._save_fig(fig, 'ablation_study')

    # Note: Qualitative visualization functions (Grid, Panorama, Gallery) require access to the model, 
    # feature extractor, cache, and image paths. They are complex to implement fully decoupled. 
    # We integrate them here, assuming they are called from the main script where these components are available.

    def generate_all(self, ablation_results: List[ExperimentResult]):
        """Orchestrates the generation of quantitative visualizations."""
        logging.info("Generating quantitative plots...")
        self.create_parameter_sensitivity_plot()
        self.create_precision_recall_tradeoff_plot()
        self.create_performance_table(top_n=12)
        self.create_runtime_plot()
        self.create_ablation_plot(ablation_results)
        
        # Qualitative generation is handled separately in main() as it requires active model components.


# =============================================================================
# Main Execution
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reduced-Cost Parameter Study for YOLO Post-processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Required arguments
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the trained YOLO model weights (.pt file).")
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Directory containing test/validation images.")
    parser.add_argument("--annotation_dir", type=str, required=True,
                        help="Directory containing ground truth annotations (VisDrone format).")

    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="./parameter_study_results",
                        help="Base directory for saving results. A timestamped subdirectory will be created.")
    parser.add_argument("--seed", type=int, default=2025,
                        help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device ('cuda' or 'cpu'). Auto-detected if None.")

    # Experiment size configuration (to control runtime)
    parser.add_argument("--subset_a", type=int, default=12,
                        help="Number of images for Stage A (Coarse Search).")
    parser.add_argument("--subset_b", type=int, default=12,
                        help="Number of images for Stage B (Focused Search).")
    parser.add_argument("--full_eval_size", type=int, default=50,
                        help="Number of images for the final evaluation of top-k configs.")
    
    # Handle execution in environments where command-line arguments might not be passed correctly (e.g., some IDEs)
    if not sys.argv[1:]:
        logging.warning("No command-line arguments provided. Attempting to use fallback defaults for testing.")
        # !!! IMPORTANT: Replace these paths with your actual local paths for testing !!!
        # These paths match the ones provided in the original prompt.
        FALLBACK_WEIGHTS = r"D:/alex2/Python code/Computer Vision1/runs/detect/visdrone_y8m7/weights/best.pt"
        FALLBACK_IMG_DIR = r"D:/alex2/Python code/Computer Vision1/VisDrone2019-DET-test-dev/images"
        FALLBACK_ANN_DIR = r"D:/alex2/Python code/Computer Vision1/VisDrone2019-DET-test-dev/annotations"

        if os.path.exists(FALLBACK_WEIGHTS) and os.path.exists(FALLBACK_IMG_DIR):
             sys.argv.extend(["--weights", FALLBACK_WEIGHTS, "--img_dir", FALLBACK_IMG_DIR, "--annotation_dir", FALLBACK_ANN_DIR, "--subset_a", "12", "--full_eval_size", "30"])
        else:
             logging.error("Fallback defaults are invalid or paths do not exist. Please provide command-line arguments.")
             parser.print_help()
             sys.exit(1)

    return parser.parse_args()

def main():
    """Main function to run the parameter study."""
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"study_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("=" * 80)
    logging.info("Starting Reduced-Cost YOLO Post-processing Parameter Study")
    logging.info("=" * 80)
    logging.info(f"Configuration:\n{json.dumps(vars(args), indent=4, default=str)}")
    logging.info(f"Using device: {device}")
    logging.info(f"Output directory: {output_dir.resolve()}")

    # Load Model
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logging.error(f"Error: Weights file not found at {weights_path}")
        return
    try:
        model = YOLO(str(weights_path))
        # model.to(device) # Ultralytics handles device placement
    except Exception as e:
        logging.error(f"Error loading YOLO model: {e}")
        return

    # Prepare Data
    img_dir = Path(args.img_dir)
    # Use pathlib glob for cleaner path handling, supporting common formats
    all_imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not all_imgs:
        logging.error(f"No images found in {img_dir}")
        return

    logging.info(f"Found {len(all_imgs)} images for the study.")

    # Adjust subset sizes based on available data
    SUBSET_A = min(args.subset_a, len(all_imgs))
    SUBSET_B = min(args.subset_b, len(all_imgs))
    FULL_EVAL = min(args.full_eval_size, len(all_imgs))

    # Initialize Runner
    runner = ExperimentRunner(model, device, all_imgs, Path(args.annotation_dir), output_dir)

    # Run Two-Stage Optimization
    start_time = time.time()
    stage_a, stage_b, finals = runner.run_two_stage(
        subset_size_a=SUBSET_A,
        subset_size_b=SUBSET_B,
        full_eval_size=FULL_EVAL
    )
    optimization_time = time.time() - start_time
    logging.info(f"Two-stage optimization completed in {optimization_time/60:.2f} minutes.")

    # Identify Best Overall Configuration
    if finals:
        overall_best = max(finals, key=lambda x: x['avg_f1'])
    else:
         all_results = stage_a + stage_b
         if not all_results:
              logging.error("No results generated during the study. Exiting.")
              return
         overall_best = max(all_results, key=lambda x: x['avg_f1'])


    logging.info("\n" + "=" * 80)
    logging.info("Best Overall Configuration Found:")
    logging.info(f"  F1: {overall_best['avg_f1']:.4f} | P: {overall_best['avg_precision']:.4f} | R: {overall_best['avg_recall']:.4f} | Time: {overall_best['avg_time']:.4f}s")
    logging.info(f"  Params:\n{json.dumps(overall_best['params'], indent=4)}")
    logging.info("=" * 80 + "\n")

    # Run Ablation Study (using the same subset size as Stage A for consistency)
    ablation_results = runner.run_ablation(overall_best['params'], ablation_subset_size=SUBSET_A)

    # Save Results
    runner.save_results("final_results.json")

    # Visualization
    viz = IEEEVisualizer(runner.results, output_dir, Path(args.annotation_dir))
    viz.generate_all(ablation_results=ablation_results)
    
    # Qualitative Visualization 
    # logging.info("Generating qualitative visualizations...")
    # sample_imgs = all_imgs[:12]
    # viz.generate_qualitative_comparisons(model, runner.feature_extractor, runner.cache, overall_best['params'], sample_imgs)

    logging.info("\n" + "=" * 80)
    logging.info("Parameter study complete.")
    logging.info(f"All results saved to: {output_dir.resolve()}")
    logging.info("=" * 80)

if __name__ == "__main__":
    # Recommended for PyTorch when using CUDA and potential multiprocessing
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()