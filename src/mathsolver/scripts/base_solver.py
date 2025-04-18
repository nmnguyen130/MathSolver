from abc import ABC, abstractmethod

class BaseSolver(ABC):
    @abstractmethod
    def can_solve(self, expr) -> bool:
        pass

    @abstractmethod
    def solve(self, expr) -> list:
        pass