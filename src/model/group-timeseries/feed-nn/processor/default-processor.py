from preprocessor import Preprocessor

class DefaultPreprocessor(Preprocessor):

    # override
    def execute(self, data):
        data = super().data
        # preprocess code
        return data