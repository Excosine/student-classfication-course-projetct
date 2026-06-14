"""
学生毕业 vs 辍学预测（二分类）—— 5 种模型对比
用法: python run_all.py
"""

import os, sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
sys.path.insert(0, CURRENT_DIR)

from preprocess import load_and_preprocess
from evaluate import Visualizer, save_results_csv
from models import train_logreg, train_rf, train_svm, train_knn, train_mlp, train_cnn


def main():
    viz = Visualizer(save_dir=os.path.join(RESULTS_DIR, "figures"))

    # ---- 预处理 ----
    print("=" * 60)
    print(">>> Step 1: Preprocessing")
    X_train, X_test, y_train, y_test, feat_names, eda = load_and_preprocess()

    # ---- EDA ----
    print("\n>>> Step 2: EDA Plots")
    viz.plot_target_distribution(list(eda["y_counts"].values()), list(eda["y_counts"].keys()))
    viz.plot_feature_distributions(eda["X_raw"])
    viz.plot_correlation_heatmap(eda["X_raw"])

    # ---- 训练（独有图在各模型内部绘制）----
    print("\n>>> Step 3: Training")
    funcs = {"LogReg": train_logreg, "RF": train_rf, "SVM": train_svm,
             "MLP": train_mlp}
    if train_knn is not None:
        funcs["KNN"] = train_knn
    if train_cnn is not None:
        funcs["CNN"] = train_cnn
    all_results = []

    for key, func in funcs.items():
        print(f"  {key} ...")
        res = func(X_train, y_train, X_test, y_test, viz=viz, feature_names=feat_names)
        all_results.append(res)
        print(f"    F1={res['f1']:.4f}  AUC={res['auc']:.4f}  Time={res['train_time']:.2f}s")

    # ---- 通用图 ----
    print("\n>>> Step 4: Per-model Basic Plots")
    for res in all_results:
        viz.plot_confusion_matrix(res["confusion_matrix"], res["model_name"])
        viz.plot_roc_curve(y_test, res["y_proba"], res["model_name"])

    # ---- 对比图 ----
    print("\n>>> Step 5: Comparison Plots")
    df = pd.DataFrame([{k: r[k] for k in ("model_name","accuracy","precision","recall","f1","auc","train_time")} for r in all_results])
    viz.plot_comparison_metrics(all_results)
    viz.plot_comparison_roc_curves(y_test, all_results)
    viz.plot_precision_recall_curves(y_test, all_results)
    viz.plot_calibration_curves(y_test, all_results)
    viz.plot_comparison_train_time(all_results)
    viz.plot_confusion_matrices_grid(all_results)

    rf_res = next((r for r in all_results if "Random Forest" in r["model_name"]), None)
    mlp_res = next((r for r in all_results if "MLP" in r["model_name"]), None)
    if rf_res and rf_res.get("model"):
        viz.plot_learning_curve(rf_res["model"], X_train, y_train, "Random Forest")
        viz.plot_pairplot_top_features(X_train, y_train, feat_names, model=rf_res["model"], model_name="RF")
    if mlp_res and mlp_res.get("model"):
        viz.plot_learning_curve(mlp_res["model"], X_train, y_train, "MLP")

    # ---- 汇总 ----
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    summary = save_results_csv(all_results, csv_path)
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(summary.to_string(index=False, float_format="%.4f"))
    print(f"\nDone. {len(all_results)} models, figures in {viz.save_dir}")


if __name__ == "__main__":
    main()
