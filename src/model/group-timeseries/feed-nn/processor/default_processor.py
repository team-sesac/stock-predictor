from .preprocessor import Preprocessor

class DefaultPreprocessor(Preprocessor):

    # override
    def execute_x(self, df):
        # preprocess code
        df = df.ffill().fillna(0)
        df = df.drop(labels=["date_id", "time_id", "row_id", "target"], axis=1)
        return df
    
    def execute_y(self, df, target):
        return df[target].ffill().fillna(0)