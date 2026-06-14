# """MLP 深度学习模型。训练后自动绘制损失曲线。"""
# import time, numpy as np
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
#
# SEED = 42
#
# def train_and_evaluate(X_train, y_train, X_test, y_test, viz=None, feature_names=None):
#     t0 = time.time()
#     model = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", early_stopping=True, max_iter=200, random_state=SEED)
#     model.fit(X_train, y_train)
#     dt = time.time() - t0
#
#     if viz is not None:
#         viz.plot_mlp_loss_curve(model)
#
#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)[:, 1]
#     return {"model_name": "MLP (Deep)", "accuracy": accuracy_score(y_test, y_pred),
#             "precision": precision_score(y_test, y_pred, zero_division=0),
#             "recall": recall_score(y_test, y_pred, zero_division=0),
#             "f1": f1_score(y_test, y_pred, zero_division=0),
#             "auc": roc_auc_score(y_test, y_proba),
#             "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
#             "y_pred": y_pred.tolist(), "y_proba": y_proba.tolist(), "train_time": dt,
#             "model": model}


"""MLP 深度学习模型。训练后自动绘制损失曲线。"""
import time, numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

SEED = 42

def train_and_evaluate(X_train, y_train, X_test, y_test, viz=None, feature_names=None):
    t0 = time.time()
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),   # 隐藏层
        activation="relu",                  # 激活函数
        solver="adam",                      # 优化器
        alpha=0.0005,                       # L2正则化系数，轻微增大防止过拟合
        batch_size=64,                      # 小批量训练，稳定收敛
        learning_rate="adaptive",           # 自适应学习率，若loss停滞则除以10
        learning_rate_init=0.001,           # 初始学习率
        max_iter=500,                       # 最大迭代次数，留足训练时间
        random_state=SEED,                  # 随机种子
        tol=1e-5,                           # 优化容忍度，让训练更充分
        early_stopping=True,                # 启用早停，防止过拟合
        validation_fraction=0.1,            # 10%训练数据作为验证集
        n_iter_no_change=15,                # 连续15轮验证损失无改善则停止
    )
    model.fit(X_train, y_train)
    dt = time.time() - t0

    if viz is not None:
        viz.plot_mlp_loss_curve(model)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {"model_name": "MLP (Deep)", "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "y_pred": y_pred.tolist(), "y_proba": y_proba.tolist(), "train_time": dt,
            "model": model}
