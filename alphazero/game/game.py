from abc import ABC, abstractmethod


class Game(ABC):

    def __init__(self):
        super().__init__()

        self.init_state = None
        self.current_state = None
        self.player = None

        # network parameters
        self.game_state_shape = None   # batch_size x channels x board_x x board_y
        self.n_conv_blocks = None
        self.n_res_blocks = None
        self.game_dim = None
        self.action_size = None

    @abstractmethod
    def encode_state(self, *args, **kwargs):
        pass

    @abstractmethod
    def decode_state(self, *args, **kwargs):
        pass

    @abstractmethod
    def view_game(self, *args, **kwargs):
        pass

    @abstractmethod
    def actions(self, *args, **kwargs):
        pass

    @abstractmethod
    def check_winner(self, *args, **kwargs):
        pass

    @abstractmethod
    def move(self, *args, **kwargs):
        pass

    @abstractmethod
    def move_back(self, *args, **kwargs):
        pass
