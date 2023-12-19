from .preprocessor import Preprocessor

class DefaultPreprocessor(Preprocessor):

    def name(self) -> str:
        return "default"

    # override
    def execute_x(self, data, target=None):
        # preprocess code
        drop_labels = ["date_id", "time_id", "row_id"]
        if target is not None:
            data = data.dropna(subset=[target])
            drop_labels.append(target)
        data = data.ffill().fillna(0)
        data = data.drop(labels=drop_labels, axis=1)
        return data
    
    def execute_y(self, data, target):
        data = data.dropna(subset=[target])
        return data[target].ffill().fillna(0)