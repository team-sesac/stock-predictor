from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Scaler():
    
    def __init__(self, scale_type: str="standard"):
        self.scale_type = scale_type
        self.__scaler_x, self.__scaler_y = self._get_scaler()
        
    def _get_scaler(self):
        if self.scale_type == "standard":
            return StandardScaler(), StandardScaler()
        elif self.scale_type == "min_max":
            return MinMaxScaler(), MinMaxScaler() 
        else:
            raise ValueError("scale_type 은 'min_max' or 'standard'만 올 수 있습니다.")
    
    def get_scaler_y(self):
        return self.__scaler_y
    
    def fit_x(self, data):
        return self.__scaler_x.fit(data)
    
    def fit_y(self, data):
        return self.__scaler_y.fit(data)
    
    def transform_x(self, data):
        return self.__scaler_x.transform(X=data)
    
    def transform_y(self, data):
        return self.__scaler_y.transform(X=data)
    
    def inverse_y(self, data):
        return self.__scaler_y.inverse_transform(data)