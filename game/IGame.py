from abc import ABC, abstractmethod
from numpy import ndarray as array


class IGame(ABC):
    @abstractmethod
    def build_state(self) -> array:
        """Build a array, use as state represenation."""
        pass

    @abstractmethod
    def handle_step(self, action_idx: int) -> int:
        """Handle the event in 1 game movement, return 1 if new session is
        initialised, return 0 otherwise."""
        pass

    @abstractmethod
    def mem_num(self) -> int:
        """Return how much time this memory should be repeated in memory pool.
        It depends on importance of this action."""
        pass

    @abstractmethod
    def update_display(self, action_idx: int, session: int,
                       epsilon: float, min_explo_rate: float,
                       memo_len: int) -> None:
        """Update the game display, pass parameters that game
        needs to visualize."""
        pass

    @abstractmethod
    def reset(self) -> bool:
        """Reset game state for each action, return bool to indicate if game
        is still running."""
        pass

    @abstractmethod
    def quit_game(self) -> None:
        """Handle clean up for game."""
        pass

    @abstractmethod
    def log_info(self, session: int) -> float:
        """Return customized log info of game of each action."""
        pass

    @property
    @abstractmethod
    def reward(self) -> float:
        """Return the reward of current step."""
        pass

    @property
    @abstractmethod
    def done(self) -> float:
        """Return if current session is finished."""
        pass

    @property
    @abstractmethod
    def action_types(self) -> int:
        """Return the number of possible actions for the game."""
        pass
