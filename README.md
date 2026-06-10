# 学生毕业 vs 辍学预测（二分类）

## 项目结构

```
project/
├── preprocess.py          # 数据预处理（独热编码 + SMOTE + 标准化）
├── evaluate.py            # 评估与可视化（混淆矩阵、ROC、对比柱状图）
├── run_all.py             # 主控脚本
├── README.md              # 本文件
├── models/
│   ├── __init__.py
│   ├── logreg.py          # 逻辑回归
│   ├── rf.py              # 随机森林
│   ├── svm.py             # 支持向量机
│   ├── knn.py             # K 近邻
│   └── mlp.py             # 多层感知机（深度学习）
└── results/               # 自动生成的结果目录
```

## 环境要求

- Python 3.10+
- 依赖安装：

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib imbalanced-learn
```

## 使用方法

### 基线模式（快速验证）

```bash
cd project
python run_all.py
```

### 优化模式（SMOTE + 超参数调优 + 集成）

```bash
cd project
python run_all.py --optimize
```

优化模式下会自动运行基线对比，生成：
- `results/results_baseline.csv` — 基线结果
- `results/results_optimized.csv` — 优化结果
- `results/comparison_baseline_vs_optimized.csv` — 优化前后对比

## 数据

数据文件 `data.csv` 请放在项目上级目录（`../data.csv`）或通过 `--data` 参数指定路径。

## 模型

| 模型 | 类型 | 超参数（优化时搜索） |
|------|------|---------------------|
| Logistic Regression | 线性 | C, solver |
| Random Forest | 集成 | n_estimators, max_depth |
| SVM | 核方法 | C, gamma |
| KNN | 距离 | n_neighbors |
| MLP | 深度学习 | hidden_layer_sizes, alpha |

优化模式额外包含 VotingClassifier 软投票集成。
