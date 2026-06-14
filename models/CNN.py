import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# 固定随机种子，保证结果可复现
def set_random_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def train_and_evaluate(X_train, y_train, X_test, y_test, viz=None, feature_names=None, class_weight=None):
    set_random_seed(42)

    if class_weight is not None:
        print(f"[CNN] 接收外部类别权重: {class_weight}")
    else:
        print("[CNN] 未传入类别权重，使用普通训练")

    # 打印全部维度，方便排错
    print(f"[CNN] X_train (训练特征) shape: {X_train.shape}")
    print(f"[CNN] y_train (训练标签) shape: {y_train.shape}")
    print(f"[CNN] X_test  (测试特征) shape: {X_test.shape}")
    print(f"[CNN] y_test  (测试标签) shape: {y_test.shape}")

    # 固定数据集特征数
    FEATURE_DIM = 155

    try:
        # --------------- 处理训练数据 ---------------
        if X_train.ndim == 2 and X_train.shape[1] == FEATURE_DIM:
            n_train = X_train.shape[0]
        else:
            n_train = y_train.shape[0]
            X_train = X_train.reshape(n_train, FEATURE_DIM)

        # --------------- 处理测试数据 ---------------
        if X_test.ndim == 2 and X_test.shape[1] == FEATURE_DIM:
            n_test = X_test.shape[0]
        else:
            n_test = y_test.shape[0]
            X_test = X_test.reshape(n_test, FEATURE_DIM)

        # 转为 1D-CNN 标准格式 (样本, 特征, 通道)
        X_train_cnn = X_train.reshape(n_train, FEATURE_DIM, 1)
        X_test_cnn = X_test.reshape(n_test, FEATURE_DIM, 1)
        print(f"[CNN] 转换完成 | 训练输入:{X_train_cnn.shape} 测试输入:{X_test_cnn.shape}")

    except Exception as e:
        print(f"[CNN] 数据维度处理失败: {str(e)}")
        raise

    # 搭建 CNN 模型
    model = Sequential(name="CNN_Student")
    model.add(Input(shape=(FEATURE_DIM, 1)))

    model.add(Conv1D(
        filters=128, kernel_size=3, activation="relu",
        padding="same", kernel_regularizer=l2(1e-4)
    ))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.32))

    model.add(Conv1D(
        filters=64, kernel_size=3, activation="relu",
        padding="same", kernel_regularizer=l2(1e-4)
    ))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.32))

    model.add(Flatten())
    model.add(Dense(80, activation="relu", kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.4))

    # 二分类输出
    model.add(Dense(1, activation="sigmoid"))

    # 模型编译
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # 回调函数
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=4, min_lr=1e-5)

    # 开始训练
    start_time = time.time()
    history = model.fit(
        X_train_cnn, y_train,
        epochs=120,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weight,
        verbose=0
    )
    train_time = time.time() - start_time

    # 预测
    y_proba = model.predict(X_test_cnn, verbose=0).ravel()
    y_pred = (y_proba >= 0.48).astype(int)

    # 计算指标
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    unique_labels = np.unique(y_test)
    if len(unique_labels) > 1:
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = 0.5

    cm = confusion_matrix(y_test, y_pred)

    # 可视化
    if viz is not None:
        try:
            model_name = "CNN"
            viz.plot_cnn_train_curve(history, model_name)
            viz.plot_cnn_prob_distribution(y_test, y_proba, model_name)
            viz.plot_cnn_error_analysis(y_test, y_pred, model_name)
        except Exception as e:
            print(f"[CNN] 可视化异常: {e}")

    # 组装返回结果
    res = {
        "model_name": "CNN",
        "model": model,
        "history": history,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "y_proba": y_proba,
        "train_time": train_time
    }
    return res