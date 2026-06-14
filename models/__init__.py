"""模型模块"""
from .logreg import train_and_evaluate as train_logreg
from .rf import train_and_evaluate as train_rf
from .svm import train_and_evaluate as train_svm
from .knn import train_and_evaluate as _train_knn
from .mlp import train_and_evaluate as train_mlp

train_knn = None  # 软删除：需要时改为 _train_knn

try:
    from .CNN import train_and_evaluate as train_cnn
except ImportError:
    train_cnn = None

