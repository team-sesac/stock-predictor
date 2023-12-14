from .preprocessor import Preprocessor

class DefaultPreprocessor(Preprocessor):

    # override
    def execute(self, data):
        # preprocess code
        return data