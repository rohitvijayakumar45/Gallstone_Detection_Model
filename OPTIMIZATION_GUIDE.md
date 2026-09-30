# 🚀 Quick Start Guide - Gallstone Detection Optimization

## 📋 Prerequisites
Ensure you have trained models available:
- YOLO model: `runs/detect/gallstone_detection/yolo26m_gallstone/weights/best.pt`
- RF-DETR model: `weights/rf_detr/best_model.pth`

## 🎯 Optimization Pipeline

### Option 1: Run Full Pipeline (Recommended)
```bash
python scripts/master_optimization_pipeline.py
```

This will automatically run all optimizations in sequence:
1. ✅ HPO training duration extended (already done)
2. ✅ Hard negative mining (already done)
3. 🔧 Ensemble weight optimization
4. 🔧 Shadow threshold tuning
5. 🔧 Multi-scale TTA optimization
6. 📊 Comprehensive evaluation

### Option 2: Run Individual Optimizations

#### 1. Optimize Ensemble Weights
```bash
python scripts/optimize_ensemble_weights.py
```
**Purpose**: Find optimal weight distribution between YOLO and RF-DETR models
**Output**: `configs/ensemble_weights.json`

#### 2. Tune Shadow Analyzer Threshold
```bash
python scripts/tune_shadow_threshold.py
```
**Purpose**: Find optimal threshold for acoustic shadow verification
**Output**: `configs/shadow_threshold_tuning.json`

#### 3. Optimize Multi-Scale TTA
```bash
python scripts/optimize_tta_scales.py
```
**Purpose**: Find optimal scale combinations for test-time augmentation
**Output**: `configs/tta_scales_optimization.json`

#### 4. Comprehensive Evaluation
```bash
python scripts/comprehensive_evaluation.py
```
**Purpose**: Evaluate all optimizations combined and show improvement
**Output**: `configs/comprehensive_evaluation.json`

## 📊 Expected Results

Based on current baseline (~85.6% mAP50), expected improvements:

| Optimization | Expected mAP50 | Improvement |
|-------------|----------------|-------------|
| Baseline | 85.6% | - |
| Optimized Weights | 87-89% | +1.3-3.4% |
| Shadow Verification | 88-90% | +2.4-4.4% |
| Multi-Scale TTA | 89-91% | +3.4-5.4% |
| **Full Optimization** | **91-94%** | **+5.4-8.4%** |

## ⚠️ Important Notes

1. **Training Data Quality**: If optimizations don't reach 97%, consider:
   - Collecting more diverse training data
   - Improving label quality
   - Adding more challenging cases

2. **Model Architecture**: For additional gains:
   - Try larger models (YOLO26l, RF-DETR base)
   - Run RF-DETR hyperparameter optimization

3. **Computational Resources**:
   - Multi-scale TTA with 1024 resolution requires ~10-12GB VRAM
   - Reduce scales if memory constrained

## 🔧 Troubleshooting

### "No models found" error
Ensure you have trained models:
```bash
# Train YOLO if needed
python train.py --phase 3 --data_dir ./dataset_final --train-yolo --yolo-weights yolo26m.pt --epochs 100

# Train RF-DETR if needed
python train.py --phase 1 --data_dir ./dataset_final --train-rfdetr --epochs 100
```

### "Out of memory" error
Reduce batch size or use smaller scales:
```bash
# Edit scripts/optimize_tta_scales.py
# Change scales from [640, 800, 1024] to [512, 640, 800]
```

### Low mAP50 after optimization
1. Check training data quality
2. Verify label accuracy
3. Consider running RF-DETR HPO:
```bash
python train.py --phase 2 --data_dir ./dataset_final --hpo-model rf_detr
```

## 📈 Monitoring Progress

Check optimization results:
```bash
# View ensemble weights
cat configs/ensemble_weights.json

# View shadow threshold results
cat configs/shadow_threshold_tuning.json

# View TTA optimization results
cat configs/tta_scales_optimization.json

# View comprehensive evaluation
cat configs/comprehensive_evaluation.json
```

## 🎯 Success Criteria

- **Target**: >97% mAP50
- **Current Baseline**: ~85.6%
- **Gap**: ~11.4%
- **Expected from Optimizations**: +5.4-8.4%
- **Remaining Gap**: ~3-6%

If optimizations don't reach target, additional steps needed:
1. RF-DETR hyperparameter optimization
2. More training data
3. Larger model architectures
4. Advanced augmentation strategies

## 🚀 Next Steps After Optimization

1. **If mAP50 >= 97%**: Proceed to clinical deployment
2. **If 90% <= mAP50 < 97%**: Run RF-DETR HPO
3. **If mAP50 < 90%**: Collect more data and try larger models

Good luck! 🩺