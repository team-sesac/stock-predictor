class EarlyStopping:
    
    def __init__(self, patience):
        self.n_triggered = 0
        self.patience = patience
        self.EARLY_STOP = False
        self.best_score = None
        
    def __call__(self, value):
        if self.best_score is None: 
            self.best_score = value
        else:
            if value <= self.best_score:
                self.n_triggered = 0
            else:
                self.n_triggered += 1
                if self.n_triggered >= self.patience:
                    self.EARLY_STOP = True
            self.best_score = value
        return self.EARLY_STOP