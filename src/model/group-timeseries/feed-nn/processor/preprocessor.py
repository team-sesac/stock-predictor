from abc import ABC, abstractmethod

class Preprocessor(ABC):

    @abstractmethod
    def execute(self, data):

        pass
