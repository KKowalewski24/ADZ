from abc import ABC, abstractmethod


class Detector(ABC):

    @abstractmethod
    def detect(self) -> None:
        pass
