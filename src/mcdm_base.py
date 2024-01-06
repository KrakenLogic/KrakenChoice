from abc import ABC, abstractmethod

class MADMBase(ABC):
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def solve(self):
        """
        Abstract method to solve the Multiple Attribute Decision Making problem.
        """
        pass

    @abstractmethod
    def report(self):
        """
        Abstract method to generate a report or summary of the decision-making process.
        """
        pass

    def get_data(self):
        """
        Get the data used for decision making.

        Returns:
        object: Data used for decision making.
        """
        return self.data

    def set_data(self, new_data):
        """
        Set the data used for decision making.

        Parameters:
        new_data (object): New data to be set.
        """
        self.data = new_data
