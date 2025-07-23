from scipy.stats import mode
import numpy as np
from sklearn.calibration import cross_val_predict
from sklearn.metrics import f1_score
from scipy.special import softmax

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def compute_oof_preds(ensemble, X, y, cls_count, cv:MultilabelStratifiedKFold):
    B, T, _ = X.shape
    oof_preds = np.zeros_like(y)
    preds_t = np.zeros(B, dtype=int)

    for train_idx, val_idx in cv.split(X, y):  # 여기 y는 (n_samples, T)
        for t in range(T):
            n_cls = cls_count[t]
            X_t = X[:, t, :]
            y_t = y[:, t]

            y_pred = ensemble.predict(X_t[val_idx])
            preds_t[val_idx] = y_pred

            oof_preds[:, t] = preds_t

    return oof_preds

class VotingEnsemble:
    def __init__(self, voting='hard', weights=None, verbose=0, num_weak_learners=None, cls_count=None, cv=None):

        if cls_count is None:
            cls_count = [2, 2, 2, 3, 2, 2]
        self.voting = voting
        self.weights = weights
        self.verbose = verbose
        self.num_weak_learners = num_weak_learners
        self.num_classes = max(cls_count) if isinstance(cls_count, list) else cls_count
        self.cls_count = cls_count

    def fit(self, X, y=None):
        # Stateless → dummy fit
        return self

    def predict(self, X):
        B = X.shape[0]
        N = self.num_weak_learners
        C = self.num_classes
        logits = X.reshape(B, N, C)

        if self.voting == 'soft':
            probs = softmax(logits, axis=-1)
            if self.weights:
                weighted_probs = probs * np.array(self.weights)[None, :, None]
                avg_probs = weighted_probs.sum(axis=1) / np.sum(self.weights)
            else:
                avg_probs = probs.mean(axis=1)
            return np.argmax(avg_probs, axis=1)

        else:
            preds = np.argmax(logits, axis=2)
            if self.weights:
                def weighted_vote(row):
                    counts = np.bincount(row, weights=self.weights, minlength=C)
                    return np.argmax(counts)
                final_preds = np.apply_along_axis(weighted_vote, 1, preds)
            else:
                final_preds, _ = mode(preds, axis=1)
                final_preds = final_preds.ravel()
            return final_preds
        
    def score(self, X, y,cv=None, mode='f1_macro'):
        """
        X: np.ndarray of shape (B, 6, max_cls * N)
        y: np.ndarray of shape (B, 6)
        cls_count: list of int, e.g., [2,2,2,3,2,2]
        cv: cross-validation strategy or int
        mode: 'f1_macro' or 'accuracy'
        """
        B, T, _ = X.shape
        if mode not in ['f1_macro', 'accuracy']:
            raise ValueError("mode must be either 'f1_macro' or 'accuracy'")

        if cv is not None:
            # Generate OOF predictions for each output
            oof_preds = compute_oof_preds(
                self,
                X, y,
                cv=cv,
                cls_count=self.cls_count
            )
            # Compute macro-F1 and accuracy for each output, then average
            f1s = [
                f1_score(y[:, i], oof_preds[:, i], average='macro')
                for i in range(y.shape[1])
            ]
            accs = [
                accuracy_score(y[:, i], oof_preds[:, i])
                for i in range(y.shape[1])
            ]


            result_dicrt = {
                'f1_macro': np.mean(f1s),
                'accuracy': np.mean(accs)
            }

            return result_dicrt[mode]

        #Should Not use cross_val_predict for VotingEnsemble
        else:
            # Direct prediction without CV
            self.fit_meta(X, y)
            y_pred = self.predict(X)
            
            result_dict = {
                'f1_macro': f1_score(y, y_pred, average='macro'),
                'accuracy': accuracy_score(y, y_pred)
            }

            return result_dict[mode]
