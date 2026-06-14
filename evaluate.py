"""
评估与可视化模块
提供 Visualizer 类：
- EDA 可视化：目标分布、特征分布、相关性热力图
- 模型独有可视化：各模型特征重要性/决策边界/损失曲线等
- 基础评估图：混淆矩阵、ROC曲线
- 对比可视化：多模型指标对比、ROC/PR/校准曲线对比
- 进阶分析：学习曲线、SHAP特征解释、特征散点矩阵
- CNN 专属可视化：训练曲线、概率分布、误差分析
"""
import os
import warnings
import numpy as np
import pandas as pd
import shutil

# ===================== 固定导入顺序（核心，绝对不能乱） =====================
# 1. 先导入matplotlib主模块
import matplotlib
# 2. 再导入绘图模块
import matplotlib.pyplot as plt
# 3. 设置无界面后端（脚本运行必备）
matplotlib.use("Agg")
# =========================================================================

# ===================== 中文乱码/方框修复（全版本兼容+强化） =====================
# 核心修复：扩展字体列表，兼容更多系统，优先保证中文显示
plt.rcParams["font.family"] = ["sans-serif"]
# 扩展中文字体列表：覆盖Windows/macOS/Linux
plt.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "SimSun", "WenQuanYi Micro Hei",
    "Heiti TC", "DejaVu Sans", "Arial Unicode MS", "PingFang SC"
]
# 强制修复负号显示为方框问题
plt.rcParams["axes.unicode_minus"] = False

# 统一图表字号，防止文字挤压重叠
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
# 自动布局，防止文字截断
plt.rcParams["figure.autolayout"] = True
# =========================================================================

# 后续依赖库（保持原有不变）
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve
from sklearn.calibration import calibration_curve

# 全局配色、样式（原有配置保留）
COLORS = sns.color_palette("Set2", n_colors=10)
CMAP_LIGHT = ListedColormap(["#FFAAAA", "#AAAAFF"])
CMAP_BOLD = ListedColormap(["#FF0000", "#0000FF"])

# 屏蔽无关警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Visualizer:
    """可视化工具类，统一管理模型评估的绘图与保存"""

    def __init__(self, save_dir="results/figures", dpi=300):
        """
        初始化可视化工具
        :param save_dir: 图片保存目录
        :param dpi: 图片分辨率
        """
        self.save_dir = os.path.abspath(save_dir)
        self.dpi = dpi
        os.makedirs(self.save_dir, exist_ok=True)

    def _save(self, filename):
        """
        统一保存图片的私有方法
        :param filename: 保存的文件名
        :return: 完整的保存路径
        """
        if not filename.endswith((".png", ".jpg")):
            filename = f"{filename}.png"

        save_path = os.path.join(self.save_dir, filename)
        try:
            plt.tight_layout()
            plt.savefig(
                save_path,
                dpi=self.dpi,
                bbox_inches="tight",
                facecolor="white",  # 确保白底
                edgecolor="none"
            )
            plt.close()
            return save_path
        except Exception as e:
            print(f"  [保存失败] {filename}: {str(e)}")
            plt.close()
            return None

    # ==================== EDA 可视化 ====================
    def plot_target_distribution(self, y_counts, labels):
        """
        绘制目标变量分布（饼图+条形图）
        :param y_counts: 各类别数量
        :param labels: 类别标签
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # 饼图
        ax1.pie(
            y_counts, labels=labels, autopct="%1.1f%%",
            colors=COLORS[:2], startangle=90, explode=(0, 0.05)
        )
        ax1.set_title("Target Distribution (Pie)")

        # 条形图
        bars = ax2.bar(labels, y_counts, color=COLORS[:2])
        for b, v in zip(bars, y_counts):
            ax2.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 3,
                str(v), ha="center", fontsize=11
            )
        ax2.set_title("Target Distribution (Bar)")
        ax2.set_ylabel("Count")

        save_path = self._save("target_distribution.png")
        if save_path:
            print(f"  [EDA] 已保存: {os.path.basename(save_path)}")

    def plot_feature_distributions(self, df, n_features=12):
        """
        绘制数值特征分布直方图（Top N）
        :param df: 特征数据框
        :param n_features: 展示的特征数量
        """
        # 筛选数值特征
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            print("  [EDA] 无数值特征可绘制")
            return

        top_features = numeric_df.columns[:min(n_features, numeric_df.shape[1])]
        rows, cols = 3, 4
        fig, axes = plt.subplots(rows, cols, figsize=(16, 10))

        for i, col in enumerate(top_features):
            ax = axes[i // cols][i % cols]
            # 去除空值后绘制直方图
            data = numeric_df[col].dropna()
            ax.hist(
                data, bins=30, color=COLORS[0],
                edgecolor="white", alpha=0.8
            )
            ax.set_title(col[:25], fontsize=8)
            ax.tick_params(labelsize=6)

        # 隐藏多余子图
        for j in range(len(top_features), rows * cols):
            axes[j // cols][j % cols].axis("off")

        plt.suptitle("Feature Distributions (Top Numeric Features)", fontsize=14)
        save_path = self._save("feature_distributions.png")
        if save_path:
            print(f"  [EDA] 已保存: {os.path.basename(save_path)}")

    def plot_correlation_heatmap(self, df, top_n=20):
        """
        绘制特征相关性热力图
        :param df: 特征数据框
        :param top_n: 展示的特征数量
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty or numeric_df.shape[1] < 2:
            print("  [EDA] 特征数量不足，无法绘制相关性热力图")
            return

        # 取Top N特征
        plot_df = numeric_df.iloc[:, :min(top_n, numeric_df.shape[1])]
        corr = plot_df.corr()

        # 绘制上三角掩码的热力图
        plt.figure(figsize=(14, 11))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, cmap="RdBu_r", center=0,
            annot=False, fmt=".2f", square=True,
            linewidths=0.3, cbar_kws={"shrink": 0.8}
        )
        plt.title("Feature Correlation Heatmap")

        save_path = self._save("correlation_heatmap.png")
        if save_path:
            print(f"  [EDA] 已保存: {os.path.basename(save_path)}")

    # ==================== 基础评估图 ====================
    def plot_confusion_matrix(self, cm, model_name):
        """
        绘制单个模型的混淆矩阵
        :param cm: 混淆矩阵
        :param model_name: 模型名称
        """
        cm = np.array(cm)
        plt.figure(figsize=(5, 4))

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Graduate", "Dropout"],
            yticklabels=["Graduate", "Dropout"]
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted"), plt.ylabel("Actual")

        # 清理模型名称中的特殊字符
        clean_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        self._save(f"cm_{clean_name}.png")

    def plot_roc_curve(self, y_test, y_proba, model_name):
        """
        绘制单个模型的ROC曲线
        :param y_test: 测试集真实标签
        :param y_proba: 预测概率
        :param model_name: 模型名称
        """
        try:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
        except Exception as e:
            print(f"  [ROC] {model_name} 绘制失败: {str(e)}")
            return

        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], "navy", lw=1, linestyle="--")
        plt.xlim([0, 1]), plt.ylim([0, 1.05])
        plt.xlabel("FPR"), plt.ylabel("TPR")
        plt.title(f"ROC - {model_name}")
        plt.legend(loc="lower right")

        clean_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        self._save(f"roc_{clean_name}.png")

    # ==================== 模型独有可视化 ====================
    def plot_logreg_coefficients(self, model, feature_names, top_n=20):
        """绘制逻辑回归系数条形图"""
        coef = model.coef_[0]
        # 取Top N重要特征
        idx = np.argsort(np.abs(coef))[-min(top_n, len(coef)):]
        vals = coef[idx]
        names = [feature_names[i] for i in idx]
        colors_bar = ["#2E86AB" if v > 0 else "#D1495B" for v in vals]

        plt.figure(figsize=(8, 6))
        plt.barh(range(len(vals)), vals, color=colors_bar)
        plt.yticks(range(len(vals)), names, fontsize=8)
        plt.axvline(0, color="black", linewidth=0.5)
        plt.xlabel("Coefficient Value")
        plt.title(f"Logistic Regression - Top {top_n} Feature Coefficients")

        save_path = self._save("logreg_coefficients.png")
        if save_path:
            print(f"  [LogReg] 已保存: {os.path.basename(save_path)}")

    def plot_rf_feature_importance(self, model, feature_names, top_n=20):
        """绘制随机森林特征重要性条形图"""
        imp = model.feature_importances_
        idx = np.argsort(imp)[-min(top_n, len(imp)):]
        vals = imp[idx]
        names = [feature_names[i] for i in idx]

        plt.figure(figsize=(8, 6))
        plt.barh(range(len(vals)), vals, color=COLORS[1])
        plt.yticks(range(len(vals)), names, fontsize=8)
        plt.xlabel("Importance")
        plt.title(f"Random Forest - Top {top_n} Feature Importances")

        save_path = self._save("rf_feature_importance.png")
        if save_path:
            print(f"  [RF] 已保存: {os.path.basename(save_path)}")

    def plot_svm_decision_boundary_pca(self, model, X_train, y_train):
        """绘制SVM决策边界（PCA降维）"""
        # PCA降维
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_train)
        explained = pca.explained_variance_ratio_

        # 重新训练2D SVM
        from sklearn.svm import SVC
        svm_2d = SVC(kernel="rbf", probability=True, random_state=42)
        svm_2d.fit(X_pca, y_train)

        # 生成网格
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # 绘图
        plt.figure(figsize=(7, 5))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=CMAP_LIGHT)
        scatter = plt.scatter(
            X_pca[:, 0], X_pca[:, 1], c=y_train,
            cmap=CMAP_BOLD, alpha=0.5, edgecolors="k", s=10
        )
        plt.xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
        plt.ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
        plt.title(f"SVM Decision Boundary (PCA, explained={sum(explained) * 100:.1f}%)")
        plt.figtext(
            0.5, 0.01,
            "Note: PCA used for visualization only; boundary not representative of original space.",
            ha="center", fontsize=7, style="italic", color="gray"
        )
        plt.legend(*scatter.legend_elements(), title="Class")

        save_path = self._save("svm_decision_boundary_pca.png")
        if save_path:
            print(f"  [SVM] 已保存: {os.path.basename(save_path)}")

    def plot_knn_decision_regions_pca(self, model, X_train, y_train):
        """绘制KNN决策区域（PCA降维）"""
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_train)
        explained = pca.explained_variance_ratio_

        # 重新训练2D KNN
        from sklearn.neighbors import KNeighborsClassifier
        knn_2d = KNeighborsClassifier(n_neighbors=model.n_neighbors)
        knn_2d.fit(X_pca, y_train)

        # 生成网格
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # 绘图
        plt.figure(figsize=(7, 5))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=CMAP_LIGHT)
        scatter = plt.scatter(
            X_pca[:, 0], X_pca[:, 1], c=y_train,
            cmap=CMAP_BOLD, alpha=0.5, edgecolors="k", s=10
        )
        plt.xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
        plt.ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
        plt.title(f"KNN (k={model.n_neighbors}) Decision Regions (PCA)")
        plt.figtext(
            0.5, 0.01,
            "Note: PCA used for visualization only; regions not representative of original space.",
            ha="center", fontsize=7, style="italic", color="gray"
        )
        plt.legend(*scatter.legend_elements(), title="Class")

        save_path = self._save("knn_decision_regions_pca.png")
        if save_path:
            print(f"  [KNN] 已保存: {os.path.basename(save_path)}")

    def plot_knn_k_values(self, X_train, y_train, X_test, y_test):
        """绘制KNN不同K值的错误率曲线"""
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

        save_path = self._save("knn_k_values.png")
        if save_path:
            print(f"  [KNN] 已保存: {os.path.basename(save_path)}")

    def plot_mlp_loss_curve(self, model):
        """绘制MLP损失曲线"""
        if not hasattr(model, "loss_curve_"):
            print("  [MLP] 无loss_curve_属性，跳过绘制")
            return

        loss = model.loss_curve_
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(loss) + 1), loss, color=COLORS[4], lw=2)
        plt.xlabel("Epoch"), plt.ylabel("Loss")
        plt.title("MLP Training Loss Curve")

        save_path = self._save("mlp_loss_curve.png")
        if save_path:
            print(f"  [MLP] 已保存: {os.path.basename(save_path)}")

    # ==================== 多模型对比可视化 ====================
    def plot_comparison_bar(self, df, metric, filename):
        """单指标对比柱状图"""
        if metric not in df.columns:
            print(f"  [对比] 无{metric}指标，跳过绘制")
            return

        plt.figure(figsize=(8, 5))
        bars = plt.bar(df["model_name"], df[metric], color=COLORS[:len(df)])
        plt.xlabel("Model"), plt.ylabel(metric.upper())
        plt.title(f"Model Comparison - {metric.upper()}")
        plt.xticks(rotation=30, ha="right")

        # 添加数值标签
        for b, v in zip(bars, df[metric]):
            plt.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{v:.4f}", ha="center", fontsize=9
            )
        plt.ylim(0, max(df[metric]) * 1.15)

        self._save(filename)

    def plot_comparison_metrics(self, results_list):
        """多模型多指标对比柱状图（Accuracy/Precision/Recall/F1/AUC）"""
        metrics = ["accuracy", "precision", "recall", "f1", "auc"]
        # 校验指标完整性
        for r in results_list:
            if not all(m in r for m in metrics):
                print(f"  [对比] 模型{r['model_name']}指标缺失，跳过多指标对比")
                return

        names = [r["model_name"] for r in results_list]
        data = {m: [r[m] for r in results_list] for m in metrics}

        x = np.arange(len(names))
        width = 0.15
        fig, ax = plt.subplots(figsize=(12, 5))

        for i, m in enumerate(metrics):
            bars = ax.bar(x + i * width, data[m], width, label=m.upper(), color=COLORS[i])
            # 添加数值标签
            for b, v in zip(bars, data[m]):
                ax.text(
                    b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=6, rotation=90
                )

        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.legend(loc="lower right")
        ax.set_title("Model Comparison - All Metrics")

        save_path = self._save("comparison_metrics.png")
        if save_path:
            print(f"  [对比] 已保存: {os.path.basename(save_path)}")

    def plot_comparison_roc_curves(self, y_test, results_list):
        """多模型ROC曲线对比"""
        plt.figure(figsize=(8, 6))

        for i, r in enumerate(results_list):
            try:
                fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
                roc_auc = auc(fpr, tpr)
                plt.plot(
                    fpr, tpr, color=COLORS[i], lw=2,
                    label=f"{r['model_name']} (AUC={roc_auc:.4f})"
                )
            except Exception as e:
                print(f"  [对比] {r['model_name']} ROC绘制失败: {str(e)}")

        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlim([0, 1]), plt.ylim([0, 1.05])
        plt.xlabel("FPR"), plt.ylabel("TPR")
        plt.title("ROC Curves - All Models")
        plt.legend(loc="lower right", fontsize=8)

        save_path = self._save("comparison_roc_curves.png")
        if save_path:
            print(f"  [对比] 已保存: {os.path.basename(save_path)}")

    def plot_comparison_train_time(self, results_list):
        """多模型训练时间对比"""
        names = [r["model_name"] for r in results_list]
        times = [r["train_time"] for r in results_list]

        plt.figure(figsize=(8, 4))
        bars = plt.bar(names, times, color=COLORS[:len(names)])
        # 添加数值标签
        for b, v in zip(bars, times):
            plt.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                f"{v:.2f}s", ha="center", fontsize=10
            )
        plt.ylabel("Time (seconds)"), plt.title("Training Time Comparison")
        plt.xticks(rotation=30, ha="right")

        save_path = self._save("comparison_train_time.png")
        if save_path:
            print(f"  [对比] 已保存: {os.path.basename(save_path)}")

    def plot_confusion_matrices_grid(self, results_list):
        """多模型混淆矩阵网格"""
        n = len(results_list)
        if n == 0:
            print("  [对比] 无模型结果，跳过混淆矩阵网格")
            return

        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))

        # 统一axes格式（适配1行/1列情况）
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = np.array([[ax] for ax in axes])

        # 绘制每个混淆矩阵
        for i, r in enumerate(results_list):
            ax = axes[i // cols][i % cols]
            cm = np.array(r["confusion_matrix"])
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Graduate", "Dropout"],
                yticklabels=["Graduate", "Dropout"]
            )
            ax.set_title(r["model_name"], fontsize=9)

        # 隐藏多余子图
        for j in range(i + 1, rows * cols):
            axes[j // cols][j % cols].axis("off")

        plt.suptitle("Confusion Matrices - All Models", fontsize=14)
        save_path = self._save("confusion_matrices_grid.png")
        if save_path:
            print(f"  [对比] 已保存: {os.path.basename(save_path)}")

    def plot_learning_curve(self, model, X_train, y_train, model_name):
        """绘制学习曲线"""
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train, cv=5, scoring="f1",
                train_sizes=np.linspace(0.1, 1.0, 10),
                random_state=42, n_jobs=-1
            )
        except Exception as e:
            print(f"  [学习曲线] {model_name} 绘制失败: {str(e)}")
            return

        train_mean = train_scores.mean(axis=1)
        val_mean = val_scores.mean(axis=1)

        plt.figure(figsize=(6, 4))
        plt.plot(train_sizes, train_mean, "o-", color=COLORS[0], label="Train F1")
        plt.plot(train_sizes, val_mean, "s-", color=COLORS[2], label="CV F1")

        # 添加误差带
        plt.fill_between(
            train_sizes,
            train_mean - train_scores.std(axis=1),
            train_mean + train_scores.std(axis=1),
            alpha=0.1, color=COLORS[0]
        )
        plt.fill_between(
            train_sizes,
            val_mean - val_scores.std(axis=1),
            val_mean + val_scores.std(axis=1),
            alpha=0.1, color=COLORS[2]
        )

        plt.xlabel("Training Size"), plt.ylabel("F1 Score")
        plt.title(f"Learning Curve - {model_name}")
        plt.legend()

        clean_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        save_path = self._save(f"learning_curve_{clean_name}.png")
        if save_path:
            print(f"  [学习曲线] 已保存: {os.path.basename(save_path)}")

    # ==================== SHAP 特征解释 ====================
    def plot_shap_summary(self, model, X_train, feature_names, model_name):
        """绘制SHAP特征重要性（Beeswarm+Bar图）"""
        try:
            import shap
        except ImportError:
            print("  [SHAP] 未安装shap库，跳过绘制 (pip install shap)")
            return

        # 采样加速（最多500样本）
        sample_size = min(500, len(X_train))
        X_sample = X_train[:sample_size]
        feature_names = list(feature_names)

        # 校验特征数量
        if len(feature_names) != X_sample.shape[1]:
            print(f"  [SHAP] 特征名称数量({len(feature_names)})与数据维度({X_sample.shape[1]})不匹配")
            return

        # 清理模型名称
        clean_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")

        # 选择合适的解释器
        try:
            if "Random Forest" in model_name or "XGBoost" in model_name:
                explainer = shap.TreeExplainer(model)
            elif "Logistic" in model_name:
                explainer = shap.LinearExplainer(model, X_sample)
            else:
                print(f"  [SHAP] 仅支持RF/LogReg/XGBoost，跳过{model_name}")
                return

            # 计算SHAP值
            exp = explainer(X_sample)
            sv = exp.values
            if sv.ndim == 3:
                sv = sv[:, :, 1]  # 取正类SHAP值

            # 绘制Beeswarm图
            shap.summary_plot(
                sv, X_sample, feature_names=feature_names,
                show=False, max_display=20
            )
            fig = plt.gcf()
            fig.set_size_inches(12, 8)
            plt.title(f"SHAP Beeswarm - {model_name}", fontsize=14)
            self._save(f"shap_beeswarm_{clean_name}.png")

            # 绘制Bar图
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
            self._save(f"shap_bar_{clean_name}.png")

            print(f"  [SHAP] 已保存: shap_beeswarm_{clean_name}.png + shap_bar_{clean_name}.png")

        except Exception as e:
            print(f"  [SHAP] {model_name} 绘制失败: {str(e)}")

    # ==================== Precision-Recall 曲线 ====================
    def plot_precision_recall_curves(self, y_test, results_list):
        """多模型PR曲线对比"""
        plt.figure(figsize=(8, 6))

        # 绘制无技能线
        no_skill = sum(y_test) / len(y_test)
        plt.axhline(
            no_skill, color="gray", lw=1, linestyle="--",
            label=f"No Skill ({no_skill:.3f})"
        )

        # 绘制各模型PR曲线
        for i, r in enumerate(results_list):
            try:
                prec, rec, _ = precision_recall_curve(y_test, r["y_proba"])
                ap = average_precision_score(y_test, r["y_proba"])
                plt.plot(
                    rec, prec, color=COLORS[i], lw=2,
                    label=f"{r['model_name']} (AP={ap:.4f})"
                )
            except Exception as e:
                print(f"  [PR] {r['model_name']} 绘制失败: {str(e)}")

        plt.xlim([0, 1]), plt.ylim([0, 1.05])
        plt.xlabel("Recall"), plt.ylabel("Precision")
        plt.title("Precision-Recall Curves - All Models")
        plt.legend(loc="lower left", fontsize=8)

        save_path = self._save("comparison_pr_curves.png")
        if save_path:
            print(f"  [PR] 已保存: {os.path.basename(save_path)}")

    # ==================== 校准曲线 ====================
    def plot_calibration_curves(self, y_test, results_list):
        """多模型校准曲线对比"""
        plt.figure(figsize=(8, 6))

        # 绘制完美校准线
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly Calibrated")

        # 绘制各模型校准曲线
        for i, r in enumerate(results_list):
            try:
                prob_true, prob_pred = calibration_curve(
                    y_test, r["y_proba"], n_bins=10, strategy="uniform"
                )
                plt.plot(
                    prob_pred, prob_true, "s-", color=COLORS[i], lw=2,
                    label=r["model_name"], markersize=5
                )
            except Exception as e:
                print(f"  [校准] {r['model_name']} 绘制失败: {str(e)}")

        plt.xlim([0, 1]), plt.ylim([0, 1])
        plt.xlabel("Mean Predicted Probability"), plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curves - All Models")
        plt.legend(loc="lower right", fontsize=8)

        save_path = self._save("comparison_calibration.png")
        if save_path:
            print(f"  [校准] 已保存: {os.path.basename(save_path)}")

    # ==================== 特征散点矩阵 ====================
    def plot_pairplot_top_features(self, X_train, y_train, feature_names,
                                   model=None, top_n=5, model_name="RF"):
        """绘制Top N特征的散点矩阵"""
        # 选择Top N特征
        if model is not None and hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            top_idx = np.argsort(imp)[-min(top_n, len(imp)):][::-1]
        else:
            top_idx = list(range(min(top_n, X_train.shape[1])))

        # 采样加速（最多1000样本）
        sample_size = min(1000, len(X_train))
        X_sub = X_train[:sample_size][:, top_idx]
        y_sub = y_train[:sample_size]

        # 构建DataFrame
        top_names = [feature_names[i] for i in top_idx]
        df = pd.DataFrame(X_sub, columns=top_names)
        df["Target"] = ["Graduate" if v == 0 else "Dropout" for v in y_sub]

        # 绘制Pairplot
        g = sns.pairplot(
            df, hue="Target", diag_kind="kde",
            palette=[COLORS[1], COLORS[2]],
            plot_kws={"alpha": 0.5, "s": 15}
        )
        g.fig.suptitle(f"Top {top_n} Feature Pairplot ({model_name})", y=1.02, fontsize=14)

        # 保存图片
        clean_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        save_path = self._save(f"pairplot_top{top_n}_{clean_name}.png")
        if save_path:
            print(f"  [Pairplot] 已保存: {os.path.basename(save_path)}")

    # ===================== CNN 专属可视化方法（修复中文显示） =====================
    def plot_cnn_train_curve(self, history, model_name="CNN"):
        """
        绘制CNN训练过程：损失 + 准确率双曲线
        :param history: Keras model.fit 返回的 history 对象
        :param model_name: 模型名称
        """
        if not history:
            print(f"  [{model_name}] 无训练历史，跳过训练曲线")
            return

        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(acc) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # 准确率曲线
        ax1.plot(epochs, acc, "b-", label="训练准确率")
        ax1.plot(epochs, val_acc, "r-", label="验证准确率")
        ax1.set_title("CNN 训练 & 验证准确率")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True)

        # 损失曲线
        ax2.plot(epochs, loss, "b-", label="训练损失")
        ax2.plot(epochs, val_loss, "r-", label="验证损失")
        ax2.set_title("CNN 训练 & 验证损失")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True)

        # 清理模型名称
        clean_name = model_name.replace(" ", "_")
        self._save(f"cnn_train_curve_{clean_name}.png")
        print(f"  [CNN] 已保存训练曲线: cnn_train_curve_{clean_name}.png")

    def plot_cnn_prob_distribution(self, y_true, y_proba, model_name="CNN"):
        """
        绘制模型预测概率分布（区分正/负样本，看分类置信度）
        """
        plt.figure(figsize=(8, 5))
        # 拆分两类样本的预测概率
        prob_0 = y_proba[y_true == 0]   # 真实: Dropout
        prob_1 = y_proba[y_true == 1]   # 真实: Graduate

        plt.hist(prob_0, bins=25, alpha=0.6, color="red", label="真实=辍学")
        plt.hist(prob_1, bins=25, alpha=0.6, color="green", label="真实=毕业")
        plt.axvline(x=0.5, color="black", linestyle="--", label="分类阈值 0.5")
        plt.xlabel("预测概率(P=毕业)")
        plt.ylabel("密度")
        plt.title(f"{model_name} 预测概率分布")
        plt.legend()
        plt.grid(True, alpha=0.3)

        clean_name = model_name.replace(" ", "_")
        self._save(f"cnn_prob_dist_{clean_name}.png")
        print(f"  [CNN] 已保存概率分布: cnn_prob_dist_{clean_name}.png")

    def plot_cnn_error_analysis(self, y_true, y_pred, y_proba, model_name="CNN"):
        """
        正误样本误差分析：优化视觉，消除密度带来的错觉
        """
        right_idx = (y_true == y_pred)
        wrong_idx = (y_true != y_pred)

        plt.figure(figsize=(8, 5))
        # 1. 移除 density=True，使用样本计数；降低红色透明度弱化干扰
        plt.hist(y_proba[right_idx], bins=25, alpha=0.7, color="#2E86AB", label="预测正确")
        plt.hist(y_proba[wrong_idx], bins=25, alpha=0.45, color="#D1495B", label="预测错误")

        # 加粗分类阈值虚线，更醒目
        plt.axvline(x=0.5, color="black", linestyle="--", linewidth=1.2, label="分类阈值 0.5")

        plt.xlabel("预测概率(P=毕业)")
        # 纵轴改为样本数量，直观体现多少条样本
        plt.ylabel("样本数量")
        plt.title(f"{model_name} 正误样本概率分布", fontsize=14)
        # 图例放到右上角，避开柱状图区域
        plt.legend(loc="upper left")
        # 网格降低透明度，不抢主体
        plt.grid(True, alpha=0.2)

        clean_name = model_name.replace(" ", "_")
        self._save(f"cnn_error_analysis_{clean_name}.png")
        print(f"  [CNN] 已保存误差分析图: cnn_error_analysis_{clean_name}.png")


# ==================== 工具函数 ====================
def save_results_csv(results_list, save_path):
    """
    保存模型评估结果到CSV文件
    :param results_list: 模型结果列表
    :param save_path: 保存路径
    :return: 结果DataFrame
    """
    if not results_list:
        print("  [保存CSV] 无结果数据")
        return pd.DataFrame()

    # 提取关键指标
    rows = []
    for r in results_list:
        rows.append({
            "model_name": r["model_name"],
            "accuracy": r.get("accuracy", np.nan),
            "precision": r.get("precision", np.nan),
            "recall": r.get("recall", np.nan),
            "f1": r.get("f1", np.nan),
            "auc": r.get("auc", np.nan),
            "train_time": r.get("train_time", np.nan)
        })

    # 排序并保存
    df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, float_format="%.4f")

    return df