class Environment(object):
    """
    The bike sharing region.
    """

    def __init__(self, delta1, delta2):
        """
        Args:
            delta1, period duration
            delta2, window duration
        """
        # game state (i.e. the state of the world)
        # every self.step() will update the state of world

        self._bikes = []  # [40, 20, 30, 50, 9], i-th number is #bikes in region i
        self._docks = []  # [40, 20, 30, 50, 9], i-th number is #docks in region i

        self._tau = 0    # current time
        self._delta1 = 0
        self._delta2 = 0
        self._reward = 0

    def step(self, action):
        # time windows
        self._rent()
        self._return()
        self._reposition(action)

    def _rent(self):
        """
        update the world state
        """
        pass

    def _return(self):
        """
        update the world state
        """
        pass

    def _reposition(self, action):
        """
        update the world state, by reposition the bikes
        Args:
            action, the action vector given by some smart guys
        """
        pass

    def reset(self):
        """
        reset the world
        """
        self._tau = 0
        pass

    @property
    def state(self):
        return self._calculate_state()

    def _calculate_state(self):
        """
        Calculate the current state
        """
        return list(zip(self._bikes, self._docks))


class Data():
    def __init__(self):
        pass

    def load(path):
        pass

    def preprocess():
        pass

    def get_trips():
        """

        """
        pass
