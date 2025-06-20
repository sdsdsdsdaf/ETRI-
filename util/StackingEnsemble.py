from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import f1_score

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
        Meta-model used in the stacking stage (e.g., lightgbm.LGBMClassifier())
    use_proba : bool, default=True
        Whether to use `predict_proba` outputs as meta-features
    """
    def __init__(self, base_models, meta_model, use_proba=True):
        self.base_models = base_models
        self.meta_model = meta_model
        self.use_proba = use_proba

    def fit_base(self, X, y):
        """
        Retrain all base models on the entire training data.
        OOF generation must be handled externally.
        """
        self.models_ = [clone(model).fit(X, y) for model in self.base_models]
        return self

    def fit_meta(self, X_meta, y):
        """
        Train the meta-model using externally generated OOF meta-features.

        Parameters
        ----------
        X_meta : array-like, shape (n_samples, n_meta_features)
            Matrix of OOF-generated meta-features
        y : array-like, shape (n_samples,)
            True target values
        """
        self.meta_model.fit(X_meta, y)
        return self

    def _build_meta_features(self, X):
        """
        Generate meta-features from trained base models for new data.
        """
        import numpy as np
        n_samples = X.shape[0]

        # Determine output dimensionality per model
        first_model = self.models_[0]
        if self.use_proba:
            sample_output = first_model.predict_proba(X[:1])
            n_outputs = sample_output.shape[1]
        else:
            n_outputs = 1

        meta_features = np.zeros((n_samples, len(self.models_) * n_outputs))
        for idx, model in enumerate(self.models_):
            if self.use_proba:
                preds = model.predict_proba(X)
            else:
                preds = model.predict(X).reshape(-1, 1)
            start = idx * n_outputs
            meta_features[:, start:start + n_outputs] = preds
        return meta_features

    def predict(self, X):
        """
        Predict class labels using the stacking ensemble.
        """
        meta_features = self._build_meta_features(X)
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X):
        """
        Predict class probabilities using the stacking ensemble.
        """
        meta_features = self._build_meta_features(X)
        return self.meta_model.predict_proba(meta_features)
        
    def score(self, X, y):
        y_pred = self.predict(X)
        return f1_score(y, y_pred, average='macro')
