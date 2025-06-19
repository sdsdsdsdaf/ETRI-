class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.counter = 0
        self.best_loss = float('inf')
        self.best_f1 = 0.0  # 추가됨
        self.best_model_state = None
        self.early_stop = False

    def __call__(self, val_loss, val_f1, model):
        improved_loss = val_loss < self.best_loss - self.min_delta and val_loss >= 0
        improved_f1 = val_f1 > self.best_f1 + self.min_delta

        if improved_loss or improved_f1:
            if improved_loss:
                self.best_loss = val_loss
            if improved_f1:
                self.best_f1 = val_f1
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_model_state:
                    model.load_state_dict(self.best_model_state)
