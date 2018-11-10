class IModel():
    """
    """
    pass


class OModel():
    """
    """

    def simulate_demands(self, period):
        """
        Args:
            period: 

        Returns:

        """
        pass


class Stimulator():
    def __init__(self, periods):
        self._om = OModel()
        self._im = IModel()
        self._periods = periods
        self._bikes = []
        self._docks = []

    def run(self, periods):
        for period in self._periods:
            self._om.simulate_demands(period)

    def rent(self, tau):
        pass

    def return_(self, tau):
        pass

    def reposition(self):
        pass


def main():
    pass


if __name__ == '__main__':
    pass
