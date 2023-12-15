from .preprocessor import Preprocessor

class DefaultPreprocessor(Preprocessor):

    # override
    def execute(self, data):
        # preprocess code
        data = data.ffill().fillna(0)
        data = data.drop(labels=["row_id"], axis=1)
        return data