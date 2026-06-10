"""逻辑回归模型。训练后自动绘制系数图 + SHAP。"""
import time, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

SEED = 42

def train_and_evaluate(X_train, y_train, X_test, y_test, viz=None, feature_names=None):
    t0 = time.time()
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    model.fit(X_train, y_train)
    dt = time.time() - t0

    if viz is not None:
        viz.plot_logreg_coefficients(model, feature_names)
        viz.plot_shap_summary(model, X_train, feature_names, "Logistic Regression")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {"model_name": "Logistic Regression", "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "y_pred": y_pred.tolist(), "y_proba": y_proba.tolist(), "train_time": dt,
            "model": model}
