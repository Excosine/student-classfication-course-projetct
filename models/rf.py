"""随机森林模型。训练后自动绘制特征重要性图 + SHAP。"""
import time, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

SEED = 42

def train_and_evaluate(X_train, y_train, X_test, y_test, viz=None, feature_names=None):
    t0 = time.time()
    model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    dt = time.time() - t0

    if viz is not None:
        viz.plot_rf_feature_importance(model, feature_names)
        viz.plot_shap_summary(model, X_train, feature_names, "Random Forest")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {"model_name": "Random Forest", "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "y_pred": y_pred.tolist(), "y_proba": y_proba.tolist(), "train_time": dt,
            "model": model}
