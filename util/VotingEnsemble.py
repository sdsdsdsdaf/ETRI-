import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from collections import Counter
import warnings

class VotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Voting ensemble for (possibly) multi-output classification.
    
    Parameters
    ----------
    base_models : list of estimator objects
        미리 인스턴스화된 sklearn 호환 분류기 리스트.
        fit(X, y) 및 predict/predict_proba를 지원해야 함.
        multi-output을 지원하는 모델이라면 y가 2D일 때도 fit 가능.
    voting : {'hard', 'soft'}, default='hard'
        'hard': 다수결 투표 (각 모델의 예측 클래스). 
        'soft': 예측 확률 평균(가중치 있으면 weighted) 후 argmax.
        soft를 쓰려면 모든 base 모델이 predict_proba를 지원해야 함.
    weights : list of float, optional
        base_models에 대응하는 가중치 리스트. None이면 동일 가중치.
    verbose : int, default=0
        0: silent, 1: info 출력
    """
    def __init__(self, base_models, voting='hard', weights=None, verbose=0):
        self.base_models = base_models
        if voting not in ('hard', 'soft'):
            raise ValueError("voting must be 'hard' or 'soft'")
        self.voting = voting
        self.weights = weights
        self.verbose = verbose

    def fit(self, X, y):
        """
        각 base 모델을 전체 학습 데이터에 대해 복제하여 학습.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        """
        # y가 1D나 2D인지 확인
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            self.multi_output_ = False
            self.n_tasks_ = 1
        elif y_arr.ndim == 2:
            self.multi_output_ = True
            self.n_tasks_ = y_arr.shape[1]
        else:
            raise ValueError("y must be 1D or 2D array for multi-output")

        # weights 설정
        if self.weights is not None:
            if len(self.weights) != len(self.base_models):
                raise ValueError("weights length must match base_models length")
            self.weights_ = np.array(self.weights, dtype=float)
        else:
            self.weights_ = np.ones(len(self.base_models), dtype=float)

        # normalize weights for soft voting (sum to 1)
        if self.voting == 'soft':
            wsum = np.sum(self.weights_)
            if wsum == 0:
                raise ValueError("Sum of weights must be > 0 for soft voting")
            self.weights_ = self.weights_ / wsum

        # fit each base model
        self.models_ = []
        for idx, model in enumerate(self.base_models):
            est = clone(model)
            if self.verbose:
                print(f"[VotingEnsemble] fitting model #{idx}: {est}")
            # multi-output 지원 여부: sklearn의 MultiOutputClassifier 등 자동 처리 가능
            est.fit(X, y)
            self.models_.append(est)
        return self

    def predict(self, X):
        """
        Hard/soft voting에 따라 최종 클래스 예측 반환.
        
        Returns
        -------
        preds : array of shape (n_samples,) or (n_samples, n_outputs)
        """
        if self.voting == 'hard':
            # 각 모델의 predict 결과를 모아 다수결 투표
            all_preds = []
            for est in self.models_:
                pred = est.predict(X)
                all_preds.append(np.asarray(pred))
            # all_preds: list of arrays, 각각 shape (n_samples,) or (n_samples, n_tasks)
            # 합치기: shape (n_models, n_samples, [n_tasks])
            stacked = np.stack(all_preds, axis=0)
            # voting
            if not self.multi_output_:
                # shape (n_models, n_samples)
                # for each sample j: majority vote among stacked[:, j]
                out = []
                for j in range(stacked.shape[1]):
                    votes = stacked[:, j]
                    cnt = Counter(votes)
                    # 다수결: 동점 시 가장 작은 클래스 선택
                    cls = min((-count, cls) for cls, count in cnt.items())[1]
                    out.append(cls)
                return np.array(out)
            else:
                # multi-output: for each task t separately
                n_samples = stacked.shape[1]
                n_tasks = stacked.shape[2]
                out = np.zeros((n_samples, n_tasks), dtype=stacked.dtype)
                for t in range(n_tasks):
                    for j in range(n_samples):
                        votes = stacked[:, j, t]
                        cnt = Counter(votes)
                        cls = min((-count, cls) for cls, count in cnt.items())[1]
                        out[j, t] = cls
                return out

        else:  # soft voting
            # 예측 확률 평균
            # for each model, get predict_proba; for multi-output, expect list or array?
            # 아래는 두 가지 케이스 처리:
            # 1) 모델.predict_proba(X) 반환이 array (n_samples, n_classes) => single-output
            # 2) 모델.predict_proba(X) 반환이 list of arrays length n_tasks => multi-output
            probas_list = []
            for est in self.models_:
                if not hasattr(est, "predict_proba"):
                    raise AttributeError(f"Estimator {est} does not support predict_proba for soft voting")
                pr = est.predict_proba(X)
                probas_list.append(pr)
            # probas_list: list of either arrays or list-of-arrays
            first = probas_list[0]
            if not self.multi_output_:
                # single-output: each pr is array (n_samples, n_classes)
                # weighted average of probabilities
                avg = None
                for w, pr in zip(self.weights_, probas_list):
                    pr_arr = np.asarray(pr)
                    if avg is None:
                        avg = w * pr_arr
                    else:
                        avg += w * pr_arr
                # 최종 예측: argmax per sample
                return np.argmax(avg, axis=1)
            else:
                # multi-output: each pr is list of arrays length n_tasks
                # 또는 array with shape (n_samples, n_tasks, ?)? 일반적으로 list.
                # 우선 list 형태 가정
                # 확인 및 표준화: probas_list[m] should be a list of length n_tasks
                n_models = len(self.models_)
                # Check structure
                if isinstance(first, list) or isinstance(first, tuple):
                    # 리스트 형태
                    # Initialize sum_proba[t] = weighted sum array shape (n_samples, n_classes_t)
                    sum_proba = []
                    for t in range(self.n_tasks_):
                        sum_proba.append(None)
                    for w, pr in zip(self.weights_, probas_list):
                        if len(pr) != self.n_tasks_:
                            raise ValueError("predict_proba output length mismatch for multi-output")
                        for t in range(self.n_tasks_):
                            arr = np.asarray(pr[t])
                            if sum_proba[t] is None:
                                sum_proba[t] = w * arr
                            else:
                                sum_proba[t] += w * arr
                    # for each task, argmax
                    n_samples = sum_proba[0].shape[0]
                    out = np.zeros((n_samples, self.n_tasks_), dtype=int)
                    for t in range(self.n_tasks_):
                        out[:, t] = np.argmax(sum_proba[t], axis=1)
                    return out
                else:
                    # 혹시 array 형태로 multi-output 확률이 담겼다면 (rare)
                    # 예: shape (n_samples, n_tasks, n_classes) — 이 경우 weighted sum 적용
                    arr0 = np.asarray(first)
                    if arr0.ndim != 3:
                        raise ValueError("Unexpected predict_proba output shape for multi-output")
                    n_samples, n_tasks, _ = arr0.shape
                    sum_proba = np.zeros_like(arr0, dtype=float)
                    for w, pr in zip(self.weights_, probas_list):
                        pr_arr = np.asarray(pr)
                        sum_proba += w * pr_arr
                    out = np.zeros((n_samples, n_tasks), dtype=int)
                    for t in range(n_tasks):
                        out[:, t] = np.argmax(sum_proba[:, t, :], axis=1)
                    return out

    def predict_proba(self, X):
        """
        soft voting 확률 평균 결과 반환.
        voting='soft'일 때만 의미 있으며, hard voting에도 사용 가능하나
        형태는 multi-output이면 list of arrays, single-output이면 array.
        """
        if self.voting != 'soft':
            warnings.warn("predict_proba is only meaningful for soft voting; returning None")
            return None

        probas_list = []
        for est in self.models_:
            if not hasattr(est, "predict_proba"):
                raise AttributeError(f"Estimator {est} does not support predict_proba")
            probas_list.append(est.predict_proba(X))

        if not self.multi_output_:
            avg = None
            for w, pr in zip(self.weights_, probas_list):
                pr_arr = np.asarray(pr)
                if avg is None:
                    avg = w * pr_arr
                else:
                    avg += w * pr_arr
            return avg  # (n_samples, n_classes)
        else:
            first = probas_list[0]
            if isinstance(first, list) or isinstance(first, tuple):
                sum_proba = [None] * self.n_tasks_
                for w, pr in zip(self.weights_, probas_list):
                    for t in range(self.n_tasks_):
                        arr = np.asarray(pr[t])
                        if sum_proba[t] is None:
                            sum_proba[t] = w * arr
                        else:
                            sum_proba[t] += w * arr
                return sum_proba  # list of arrays, each (n_samples, n_classes_t)
            else:
                arr0 = np.asarray(first)
                n_samples, n_tasks, _ = arr0.shape
                sum_proba = np.zeros_like(arr0, dtype=float)
                for w, pr in zip(self.weights_, probas_list):
                    sum_proba += w * np.asarray(pr)
                # return array shape (n_samples, n_tasks, n_classes)
                return sum_proba

    def set_verbose(self, verbose):
        """verbose level 설정"""
        self.verbose = verbose
        # 내부 base 모델에도 verbose 속성이 있으면 설정 시도
        for est in self.models_:
            if hasattr(est, 'set_params'):
                try:
                    est.set_params(verbose=verbose)
                except Exception:
                    pass

    def get_verbose(self):
        return self.verbose