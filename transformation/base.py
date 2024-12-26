

class Transformator(ABC):
    @abstractmethod
    def transform(self, dataset: list[Audio], labels: list[bool]):
        pass
