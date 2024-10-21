from abc import ABC, abstractmethod


class BaseNlp(ABC):
    @abstractmethod
    def get_doc(self, text: str):
        pass

    @abstractmethod
    def resolve_coreferences(self, text: str) -> str:
        pass

    @abstractmethod
    def convert_third_person_to_first_person(self, text: str, name: str) -> str:
        pass

    @abstractmethod
    def convert_third_person_to_instruction(self, text: str, name: str) -> str:
        pass
