import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from typing import Optional, Union

from util.TaskSpecificEnsemble import TaskSpecificEnsemble

class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Stacking ensemble wrapper class with externalized OOF feature generation.

    - `fit_base(X, y)`: Retrain base models on the full dataset
    - `fit_meta(X_meta, y)`: Train the meta-model using externally provided OOF meta-features
    - `predict` / `predict_proba`: Generate final predictions using trained base and meta models

    Parameters
    ----------
    base_models : list
        List of pre-instantiated scikit-learn compatible base model objects
    meta_model : estimator
        Base estimator for the meta stage; will be wrapped for multi-output
    use_proba : bool, default=True
        Whether to use `predict_proba` outputs as meta-features
    """
    def __init__(self, base_models=None, meta_model=None, use_proba=True, cls_count=None):
        if meta_model is None:
            raise ValueError("meta_model must be provided")
        if cls_count is None:
            cls_count = [2, 2, 2, 3, 2, 2]
        
        self.cls_count = cls_count
        self.base_models = base_models
        # Wrap the provided meta_model into a multi-output classifier
        self.meta_model = TaskSpecificEnsemble(meta_model, cls_count=cls_count)
        self.use_proba = use_proba

    def fit_base(self, X, y):
        """
        Retrain all base models on the entire training data.
        OOF generation must be handled externally.
        """
        # Clone and fit each base model on full data
        self.models_ = [clone(model).fit(X, y) for model in self.base_models]
        return self

    def fit_meta(self, X_meta, y):
        """
        Train the wrapped multi-output meta-model on OOF features.

        Parameters
        ----------
        X_meta : array-like, shape (n_samples, n_meta_features)
            Matrix of OOF-generated meta-features
        y : array-like, shape (n_samples, n_outputs)
            True multi-output target values
        """
        # Fit the multi-output meta_model directly
        self.meta_model.fit(X_meta, y)
        return self

    def predict(self, X):
        """
        Predict class labels using the stacking ensemble.
        """
        # Delegate to multi-output classifier
        return self.meta_model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities using the stacking ensemble.
        """
        # Delegate probability prediction
        return self.meta_model.predict_proba(X)

    def score(self, X, y, cv=None, mode='f1_macro'):
        """
        Compute both macro-F1 and accuracy of the stacking ensemble.

        If `cv` is provided, generate out-of-fold predictions
        for each output and compute their average metrics.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_meta_features)
            Meta-features (base-model OOF predictions)
        y : array-like, shape (n_samples, n_outputs)
            True multi-output labels
        cv : int or cross-validation generator, optional
            If given, run cross_val_predict over the wrapped meta-model
            and compute metrics on OOF preds.

        mode : str, default='f1_macro'
            Metric to return, either 'f1_macro' or 'accuracy'.

        Returns
        -------
        dict
            Dictionary with keys 'f1_macro' and 'accuracy'.
        """

        if mode not in ['f1_macro', 'accuracy']:
            raise ValueError("mode must be either 'f1_macro' or 'accuracy'")

        if cv is not None:
            # Generate OOF predictions for each output
            oof_preds = cross_val_predict(
                self.meta_model,
                X, y,
                cv=cv,
                method='predict'
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
        else:
            # Direct prediction without CV
            self.fit_meta(X, y)
            y_pred = self.predict(X)
            
            result_dict = {
                'f1_macro': f1_score(y, y_pred, average='macro'),
                'accuracy': accuracy_score(y, y_pred)
            }

            return result_dict[mode]
