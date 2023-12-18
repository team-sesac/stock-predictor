from abc import ABC, abstractmethod

class Preprocessor(ABC):

    @abstractmethod
    def execute_x(self, data):
        pass
    
    @abstractmethod
    def execute_y(self, data, target):
        pass
