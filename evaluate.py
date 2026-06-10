"""
评估与可视化模块
提供 Visualizer 类：EDA / 模型独有图 / 对比图 / 优化对比图 / 基础评估图。
"""

import os
import matplotlib
matplotlib.use("Agg")  # 非交互后端，避免 tkinter 错误

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import learning_curve
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, average_precision_score

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

COLORS = sns.color_palette("Set2", n_colors=10)
CMAP_LIGHT = ListedColormap(["#FFAAAA", "#AAAAFF"])
CMAP_BOLD = ListedColormap(["#FF0000", "#0000FF"])


class Visualizer:

    def __init__(self, save_dir="results/figures", dpi=300):
        self.save_dir = save_dir
        self.dpi = dpi
        os.makedirs(save_dir, exist_ok=True)

    def _save(self, name):
        path = os.path.join(self.save_dir, name)
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        return path

    # ==================== EDA ====================

    def plot_target_distribution(self, y_counts, labels):
        """饼图 + 条形图：目标变量分布"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.pie(y_counts, labels=labels, autopct="%1.1f%%", colors=COLORS[:2],
                startangle=90, explode=(0, 0.05))
        ax1.set_title("Target Distribution (Pie)")
        bars = ax2.bar(labels, y_counts, color=COLORS[:2])
        for b, v in zip(bars, y_counts):
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 3, str(v),
                     ha="center", fontsize=11)
        ax2.set_title("Target Distribution (Bar)")
        ax2.set_ylabel("Count")
        self._save("target_distribution.png")
        print(f"  [EDA] target_distribution.png")

    def plot_feature_distributions(self, df, n_features=12):
        """前 n 个数值特征的分布直方图（3x4 子图）"""
        numeric = df.select_dtypes(include=[np.number]).columns[:n_features]
        rows, cols = 3, 4
        fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
        for i, col in enumerate(numeric):
            ax = axes[i // cols][i % cols]
            ax.hist(df[col].dropna(), bins=30, color=COLORS[0], edgecolor="white", alpha=0.8)
            ax.set_title(col[:25], fontsize=8)
            ax.tick_params(labelsize=6)
        for j in range(i + 1, rows * cols):
            axes[j // cols][j % cols].axis("off")
        plt.suptitle("Feature Distributions (Top 12 Numeric Features)", fontsize=14)
        self._save("feature_distributions.png")
        print(f"  [EDA] feature_distributions.png")

    def plot_correlation_heatmap(self, df, top_n=20):
        """特征相关性热力图（top_n 个数值特征）"""
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] > top_n:
            numeric = numeric.iloc[:, :top_n]
        corr = numeric.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.figure(figsize=(14, 11))
        sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, annot=False,
                    fmt=".2f", square=True, linewidths=0.3,
                    cbar_kws={"shrink": 0.8})
        plt.title("Feature Correlation Heatmap")
        self._save("correlation_heatmap.png")
        print(f"  [EDA] correlation_heatmap.png")

    # ==================== 基础评估图 ====================

    def plot_confusion_matrix(self, cm, model_name):
        """单个模型混淆矩阵热力图"""
        cm = np.array(cm)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Graduate", "Dropout"],
                    yticklabels=["Graduate", "Dropout"])
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted"), plt.ylabel("Actual")
        name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        self._save(f"cm_{name}.png")

    def plot_roc_curve(self, y_test, y_proba, model_name):
        """单个模型 ROC 曲线"""
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], "navy", lw=1, linestyle="--")
        plt.xlim([0, 1]), plt.ylim([0, 1.05])
        plt.xlabel("FPR"), plt.ylabel("TPR")
        plt.title(f"ROC - {model_name}")
        plt.legend(loc="lower right")
        name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        self._save(f"roc_{name}.png")

    # ==================== 模型独有图 ====================

    def plot_logreg_coefficients(self, model, feature_names, top_n=20):
        """逻辑回归系数水平条形图（前 top_n 个最重要特征）"""
        coef = model.coef_[0]
        idx = np.argsort(np.abs(coef))[-top_n:]
        vals = coef[idx]
        names = [feature_names[i] for i in idx]
        colors_bar = ["#2E86AB" if v > 0 else "#D1495B" for v in vals]

        plt.figure(figsize=(8, 6))
        plt.barh(range(len(vals)), vals, color=colors_bar)
        plt.yticks(range(len(vals)), names, fontsize=8)
        plt.axvline(0, color="black", linewidth=0.5)
        plt.xlabel("Coefficient Value")
        plt.title(f"Logistic Regression - Top {top_n} Feature Coefficients")
        self._save("logreg_coefficients.png")
        print(f"  [LogReg] logreg_coefficients.png")

    def plot_rf_feature_importance(self, model, feature_names, top_n=20):
        """随机森林特征重要性水平条形图（前 top_n）"""
        imp = model.feature_importances_
        idx = np.argsort(imp)[-top_n:]
        vals = imp[idx]
        names = [feature_names[i] for i in idx]

        plt.figure(figsize=(8, 6))
        plt.barh(range(len(vals)), vals, color=COLORS[1])
        plt.yticks(range(len(vals)), names, fontsize=8)
        plt.xlabel("Importance")
        plt.title(f"Random Forest - Top {top_n} Feature Importances")
        self._save("rf_feature_importance.png")
        print(f"  [RF] rf_feature_importance.png")

    def plot_svm_decision_boundary_pca(self, model, X_train, y_train):
        """SVM 决策边界（PCA 降维到 2D）"""
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_train)
        explained = pca.explained_variance_ratio_

        # 只在 PCA 空间重新训练 SVM
        from sklearn.svm import SVC
        svm_2d = SVC(kernel="rbf", probability=True, random_state=42)
        svm_2d.fit(X_pca, y_train)

        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        plt.figure(figsize=(7, 5))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=CMAP_LIGHT)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap=CMAP_BOLD,
                              alpha=0.5, edgecolors="k", s=10)
        plt.xlabel(f"PC1 ({explained[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({explained[1]*100:.1f}%)")
        plt.title(f"SVM Decision Boundary (PCA, explained={sum(explained)*100:.1f}%)")
        plt.figtext(0.5, 0.01, "Note: PCA used for visualization only; boundary not representative of original 155-D space.",
                    ha="center", fontsize=7, style="italic", color="gray")
        legend1 = plt.legend(*scatter.legend_elements(), title="Class")
        plt.gca().add_artist(legend1)
        self._save("svm_decision_boundary_pca.png")
        print(f"  [SVM] svm_decision_boundary_pca.png")

    def plot_knn_decision_regions_pca(self, model, X_train, y_train):
        """KNN 决策区域（PCA 降维到 2D）"""
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_train)
        explained = pca.explained_variance_ratio_

        from sklearn.neighbors import KNeighborsClassifier
        knn_2d = KNeighborsClassifier(n_neighbors=model.n_neighbors)
        knn_2d.fit(X_pca, y_train)

        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        plt.figure(figsize=(7, 5))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=CMAP_LIGHT)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap=CMAP_BOLD,
                              alpha=0.5, edgecolors="k", s=10)
        plt.xlabel(f"PC1 ({explained[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({explained[1]*100:.1f}%)")
        plt.title(f"KNN (k={model.n_neighbors}) Decision Regions (PCA)")
        plt.figtext(0.5, 0.01, "Note: PCA used for visualization only; regions not representative of original 155-D space.",
                    ha="center", fontsize=7, style="italic", color="gray")
        legend1 = plt.legend(*scatter.legend_elements(), title="Class")
        plt.gca().add_artist(legend1)
        self._save("knn_decision_regions_pca.png")
        print(f"  [KNN] knn_decision_regions_pca.png")

    def plot_knn_k_values(self, X_train, y_train, X_test, y_test):
        """不同 K 值的错误率变化折线图"""
        ks = range(1, 20, 2)
        errors = []
        from sklearn.neighbors import KNeighborsClassifier
        for k in ks:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            errors.append(1 - accuracy_score(y_test, knn.predict(X_test)))
        plt.figure(figsize=(6, 4))
        plt.plot(ks, errors, "o-", color=COLORS[3], markersize=6)
        plt.xlabel("K"), plt.ylabel("Error Rate")
        plt.title("KNN - Error Rate vs K")
        plt.xticks(ks)
        self._save("knn_k_values.png")
        print(f"  [KNN] knn_k_values.png")

    def plot_mlp_loss_curve(self, model):
        """MLP 训练损失曲线"""
        if not hasattr(model, "loss_curve_"):
            print("  [MLP] No loss_curve_ available, skipping.")
            return
        loss = model.loss_curve_
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(loss) + 1), loss, color=COLORS[4], lw=2)
        plt.xlabel("Epoch"), plt.ylabel("Loss")
        plt.title("MLP Training Loss Curve")
        self._save("mlp_loss_curve.png")
        print(f"  [MLP] mlp_loss_curve.png")

    # ==================== 对比图 ====================

    def plot_comparison_bar(self, df, metric, filename):
        """单指标对比柱状图"""
        plt.figure(figsize=(8, 5))
        bars = plt.bar(df["model_name"], df[metric], color=COLORS[:len(df)])
        plt.xlabel("Model"), plt.ylabel(metric.upper())
        plt.title(f"Model Comparison - {metric.upper()}")
        plt.xticks(rotation=30, ha="right")
        for b, v in zip(bars, df[metric]):
            plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                     f"{v:.4f}", ha="center", fontsize=9)
        plt.ylim(0, max(df[metric]) * 1.15)
        self._save(filename)

    def plot_comparison_metrics(self, results_list):
        """五指标分组柱状图（Accuracy, Precision, Recall, F1, AUC）"""
        metrics = ["accuracy", "precision", "recall", "f1", "auc"]
        names = [r["model_name"] for r in results_list]
        data = {m: [r[m] for r in results_list] for m in metrics}

        x = np.arange(len(names))
        width = 0.15
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, m in enumerate(metrics):
            bars = ax.bar(x + i * width, data[m], width, label=m.upper(), color=COLORS[i])
            for b, v in zip(bars, data[m]):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                        f"{v:.3f}", ha="center", fontsize=6, rotation=90)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.legend(loc="lower right")
        ax.set_title("Model Comparison - All Metrics")
        self._save("comparison_metrics.png")
        print(f"  [Compare] comparison_metrics.png")

    def plot_comparison_roc_curves(self, y_test, results_list):
        """所有模型 ROC 曲线叠加"""
        plt.figure(figsize=(8, 6))
        for i, r in enumerate(results_list):
            fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=COLORS[i], lw=2,
                     label=f"{r['model_name']} (AUC={roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlim([0, 1]), plt.ylim([0, 1.05])
        plt.xlabel("FPR"), plt.ylabel("TPR")
        plt.title("ROC Curves - All Models")
        plt.legend(loc="lower right", fontsize=8)
        self._save("comparison_roc_curves.png")
        print(f"  [Compare] comparison_roc_curves.png")

    def plot_comparison_train_time(self, results_list):
        """训练时间对比柱状图"""
        names = [r["model_name"] for r in results_list]
        times = [r["train_time"] for r in results_list]
        plt.figure(figsize=(8, 4))
        bars = plt.bar(names, times, color=COLORS[:len(names)])
        for b, v in zip(bars, times):
            plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                     f"{v:.2f}s", ha="center", fontsize=10)
        plt.ylabel("Time (seconds)"), plt.title("Training Time Comparison")
        plt.xticks(rotation=30, ha="right")
        self._save("comparison_train_time.png")
        print(f"  [Compare] comparison_train_time.png")

    def plot_confusion_matrices_grid(self, results_list):
        """混淆矩阵网格（2x3 子图，最多 6 个模型）"""
        n = len(results_list)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = np.array([[ax] for ax in axes])

        for i, r in enumerate(results_list):
            ax = axes[i // cols][i % cols]
            cm = np.array(r["confusion_matrix"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Graduate", "Dropout"],
                        yticklabels=["Graduate", "Dropout"])
            ax.set_title(r["model_name"], fontsize=9)
        for j in range(i + 1, rows * cols):
            axes[j // cols][j % cols].axis("off")
        plt.suptitle("Confusion Matrices - All Models", fontsize=14)
        self._save("confusion_matrices_grid.png")
        print(f"  [Compare] confusion_matrices_grid.png")

    def plot_learning_curve(self, model, X_train, y_train, model_name):
        """学习曲线（训练集大小 vs 交叉验证得分）"""
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=5, scoring="f1",
            train_sizes=np.linspace(0.1, 1.0, 10), random_state=42, n_jobs=-1,
        )
        train_mean = train_scores.mean(axis=1)
        val_mean = val_scores.mean(axis=1)

        plt.figure(figsize=(6, 4))
        plt.plot(train_sizes, train_mean, "o-", color=COLORS[0], label="Train F1")
        plt.plot(train_sizes, val_mean, "s-", color=COLORS[2], label="CV F1")
        plt.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                         train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color=COLORS[0])
        plt.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                         val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1, color=COLORS[2])
        plt.xlabel("Training Size"), plt.ylabel("F1 Score")
        plt.title(f"Learning Curve - {model_name}")
        plt.legend()
        name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        self._save(f"learning_curve_{name}.png")
        print(f"  [Learn] learning_curve_{name}.png")

    # ==================== SHAP 特征重要性 ====================

    def plot_shap_summary(self, model, X_train, feature_names, model_name):
        """SHAP Beeswarm + Bar 图（仅 RF / LogReg，需安装 shap）"""
        try:
            import shap
        except ImportError:
            print("  [SHAP] shap not installed, skipping. pip install shap")
            return

        X_sample = X_train[:min(500, len(X_train))]
        feature_names = list(feature_names)  # 确保是 list
        assert len(feature_names) == X_sample.shape[1]
        name_key = model_name.replace(" ", "_").replace("(", "").replace(")", "")

        if "Random Forest" in model_name or "XGBoost" in model_name:
            explainer = shap.TreeExplainer(model)
        elif "Logistic" in model_name:
            explainer = shap.LinearExplainer(model, X_sample)
        else:
            # KernelExplainer 在 155 维特征上过慢，仅 RF/LogReg 支持 SHAP
            print(f"  [SHAP] Skipping {model_name} (only RF/LogReg supported for speed)")
            return

        # 使用新 API explainer(X) 获取正确形状 (n_samples, n_features, n_classes)
        exp = explainer(X_sample)
        sv = exp.values  # shape (n_samples, n_features, n_classes)
        if sv.ndim == 3:
            sv = sv[:, :, 1]  # 取正类 (Dropout) 的 SHAP 值

        # ---- Beeswarm（shap.summary_plot 旧 API，兼容性好）----
        shap.summary_plot(sv, X_sample, feature_names=feature_names,
                          show=False, max_display=20)
        fig = plt.gcf()
        fig.set_size_inches(12, 8)
        ax = plt.gca()
        ax.set_title(f"SHAP Beeswarm - {model_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"shap_beeswarm_{name_key}.png"),
                    dpi=self.dpi, bbox_inches="tight")
        plt.close("all")

        # ---- Bar（自定义 matplotlib，可控且清晰）----
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx = np.argsort(mean_abs)[-20:]
        top_vals = mean_abs[top_idx]
        top_names = [feature_names[int(i)] for i in top_idx]

        plt.figure(figsize=(10, 7))
        sv_top_mean = sv[:, top_idx.astype(int)].mean(axis=0)
        colors = ["#2E86AB" if v > 0 else "#D1495B" for v in sv_top_mean]
        plt.barh(range(20), top_vals, color=colors)
        plt.yticks(range(20), top_names, fontsize=9)
        plt.xlabel("Mean |SHAP| Value", fontsize=12)
        plt.title(f"SHAP Feature Importance - {model_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"shap_bar_{name_key}.png"),
                    dpi=self.dpi, bbox_inches="tight")
        plt.close("all")
        print(f"  [SHAP] shap_beeswarm_{name_key}.png + shap_bar_{name_key}.png")

    # ==================== Precision-Recall 曲线 ====================

    def plot_precision_recall_curves(self, y_test, results_list):
        """所有模型的 Precision-Recall 曲线叠加"""
        plt.figure(figsize=(8, 6))
        for i, r in enumerate(results_list):
            prec, rec, _ = precision_recall_curve(y_test, r["y_proba"])
            ap = average_precision_score(y_test, r["y_proba"])
            plt.plot(rec, prec, color=COLORS[i], lw=2,
                     label=f"{r['model_name']} (AP={ap:.4f})")
        no_skill = sum(y_test) / len(y_test)
        plt.axhline(no_skill, color="gray", lw=1, linestyle="--",
                    label=f"No Skill ({no_skill:.3f})")
        plt.xlim([0, 1]), plt.ylim([0, 1.05])
        plt.xlabel("Recall"), plt.ylabel("Precision")
        plt.title("Precision-Recall Curves - All Models")
        plt.legend(loc="lower left", fontsize=8)
        self._save("comparison_pr_curves.png")
        print(f"  [PR] comparison_pr_curves.png")

    # ==================== 校准曲线 ====================

    def plot_calibration_curves(self, y_test, results_list):
        """所有模型的校准曲线（预测概率 vs 实际频率）"""
        plt.figure(figsize=(8, 6))
        for i, r in enumerate(results_list):
            prob_true, prob_pred = calibration_curve(
                y_test, r["y_proba"], n_bins=10, strategy="uniform"
            )
            plt.plot(prob_pred, prob_true, "s-", color=COLORS[i], lw=2,
                     label=r["model_name"], markersize=5)
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly Calibrated")
        plt.xlim([0, 1]), plt.ylim([0, 1])
        plt.xlabel("Mean Predicted Probability"), plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curves - All Models")
        plt.legend(loc="lower right", fontsize=8)
        self._save("comparison_calibration.png")
        print(f"  [Calib] comparison_calibration.png")

    # ==================== Pairplot（top 特征散点矩阵）====================

    def plot_pairplot_top_features(self, X_train, y_train, feature_names,
                                   model=None, top_n=5, model_name="RF"):
        """Top N 最重要特征的散点矩阵（按 RF 重要性降序取前 N 个连续特征）"""
        if model is not None and hasattr(model, "feature_importances_"):
            # 用 RF 特征重要性选 top 连续特征
            imp = model.feature_importances_
            top_idx = np.argsort(imp)[-top_n:][::-1]
        else:
            top_idx = list(range(min(top_n, X_train.shape[1])))

        top_names = [feature_names[i] for i in top_idx]
        X_sub = X_train[:min(1000, len(X_train))][:, top_idx]

        df = pd.DataFrame(X_sub, columns=top_names)
        df["Target"] = ["Graduate" if v == 0 else "Dropout" for v in
                         y_train[:min(1000, len(y_train))]]

        g = sns.pairplot(df, hue="Target", diag_kind="kde",
                         palette=[COLORS[1], COLORS[2]],
                         plot_kws={"alpha": 0.5, "s": 15})
        g.fig.suptitle(f"Top {top_n} Feature Pairplot ({model_name})", y=1.02, fontsize=14)
        name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        self._save(f"pairplot_top{top_n}_{name}.png")
        print(f"  [Pairplot] pairplot_top{top_n}_{name}.png")

# ---- 工具函数 ----

def save_results_csv(results_list, save_path):
    rows = []
    for r in results_list:
        rows.append({"model_name": r["model_name"], "accuracy": r["accuracy"],
                     "precision": r["precision"], "recall": r["recall"],
                     "f1": r["f1"], "auc": r["auc"], "train_time": r["train_time"]})
    df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df
