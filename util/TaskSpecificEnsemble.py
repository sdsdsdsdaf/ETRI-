import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

class TaskSpecificEnsemble(BaseEstimator, ClassifierMixin):
    """
    여러 태스크용 메타 모델을 태스크별로 다른 피처 X_task에 맞춰 학습/예측하는 wrapper.
    base_estimator는 단일 태스크용 추정기(e.g. LGBMClassifier).
    """

    def __init__(self, base_estimator, cls_count):
        self.base_estimator = base_estimator
        self.cls_count = cls_count

    def fit(self, X_list:np.ndarray, y):
        """
        X_list: list/tuple of length n_tasks, each element은 (n_samples, n_feat_t) ndarray
        y: array-like of shape (n_samples, n_tasks)
        """
        n_tasks = len(self.cls_count)
        self.models_ = []
        for t in range(n_tasks):
            Xt = X_list[:, t, :self.cls_count[t]].reshape(-1,self.cls_count[t])
            yt = y[:, t]    
            est = clone(self.base_estimator)
            est.fit(Xt, yt)
            self.models_.append(est)
        return self

    def predict(self, X_list:np.ndarray):
        """
        X_list: list of length n_tasks, 각 (n_samples, n_feat_t)
        반환: array of shape (n_samples, n_tasks)
        """
        n_tasks = len(self.cls_count)
        preds = []
        for t in range(n_tasks):
            Xt = X_list[:, t, :self.cls_count[t]].reshape(-1,self.cls_count[t])
            est = self.models_[t]
            preds_t = est.predict(Xt)
            preds.append(preds_t.reshape(-1, 1))

        return np.hstack(preds)

    def predict_proba(self, X_list):
        """
        X_list: list of length n_tasks
        반환: list of arrays, 각 (n_samples, n_classes_t)
        """
        probas = []
        for t in range(len(X_list)):
            Xt = X_list[t]
            est = self.models_[t]
            probas.append(est.predict_proba(Xt))
        return probas