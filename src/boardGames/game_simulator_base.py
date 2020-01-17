from abc import ABC, abstractmethod

class GameSimulator(ABC):
    @property
    @abstractmethod
    def is_approximate_simulator(self):
        """
        Returns True if this simulator is not the real one
        """

    @abstractmethod
    def actions(self, state, player:int):
        """
        Returns the list of possible actions for the player in the current state
        (state, player) -> List[actions]
        """

    @abstractmethod
    def result(self, state, player:int, action):
        """
        Returns the next state and next player from a player's action
        (state, player, action) -> state, player
        """

    @abstractmethod
    def results(self, states, player:int, actions, rnd):
        """
        Returns the n next states for n actions
        (states, player, actions) -> states, player
        """

    def result_debug(self, state, player:int, action):
        pass

    @abstractmethod
    def terminal_test(self, state):
        """
        Returns True if the game is over
        (state) -> bool
        """

    @abstractmethod
    def utility(self, state, player:int):
        """
        Returns the utility of this state for this player
        (1 if victory, -1 if defeat or draw, 0 for draw)
        (state, player) -> int
        """

    @abstractmethod
    def max_turns(self, state):
        """
        Returns the number of turns that could still occur from this state (to avoid loops)
        (state) -> int
        """

    @abstractmethod
    def get_state_data(self, state):
        """
        Returns the state in a parsable format
        (state) -> List[int]
        """

    @abstractmethod
    def get_action_data(self, state, player:int, action):
        """
        Returns the action in a parsable format
        (player, action) -> List[int]
        """

    @abstractmethod
    def get_initial_state(self, starting_player:int):
        """
        Returns the game's initial state
        (starting_player) -> state
        """

    @staticmethod
    def get_state_data_len(board_dim:int):
        """
        Return state data length
        (int (just one axis)) -> int
        """

    @staticmethod
    def get_action_data_len(board_dim:int):
        """
        Return action data length
        (int (just one axis)) -> int
        """

    @abstractmethod
    def get_state_from_data(self, last_state, data):
        """
        Return the state from data
        """