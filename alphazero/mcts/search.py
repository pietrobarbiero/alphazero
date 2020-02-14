from .node import MCTSNode, DummyNode
from ..game.game import Game
import torch


def mcts(game: Game, num_reads, net, temp, n_children):
    cuda = torch.cuda.is_available() 
    root = MCTSNode(game, move=None, n_children=n_children, parent=DummyNode())
    for i in range(num_reads):
        leaf = root.select_leaf()
        encoded_s = game.encode_state()
        encoded_s = encoded_s.transpose(2, 0, 1)
        if cuda:
            encoded_s = torch.from_numpy(encoded_s).float().cuda()
        else:
            encoded_s = torch.from_numpy(encoded_s).float()
        child_priors, value_estimate = net(encoded_s)
        child_priors = child_priors.detach().cpu().numpy().reshape(-1)
        value_estimate = value_estimate.item()
        if leaf.game.check_winner() is True or leaf.game.actions() == []:  # if somebody won or draw
            leaf.backup(value_estimate)
            continue
        leaf.expand(child_priors)  # need to make sure valid moves
        leaf.backup(value_estimate)
    return root
