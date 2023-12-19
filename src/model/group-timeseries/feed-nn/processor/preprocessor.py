from abc import ABC, abstractmethod

class Preprocessor(ABC):

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def execute_x(self, data, target=None):
        pass
    
    @abstractmethod
    def execute_y(self, data, target):
        pass
