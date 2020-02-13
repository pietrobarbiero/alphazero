from .connect4.game import Connect4
from .net.net import AlphaNet
from .net.learning import learn
from .net.evaluating import evaluate
from .mcts.play import self_play, run_MCTS
# from ._alpha_net import ConnectNet, AlphaLoss
# from ._mcts import run_MCTS

from ._version import __version__

__all__ = ['Connect4', 'AlphaNet', 'self_play',
           # 'ConnectNet', 'AlphaLoss',
           'run_MCTS', 'learn', 'evaluate',
           '__version__']
