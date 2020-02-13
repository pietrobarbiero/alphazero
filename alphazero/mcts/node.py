import collections
import numpy as np
import math
import copy
import torch
import torch.multiprocessing as mp
import datetime
import logging
from tqdm import tqdm
from anytree import NodeMixin, RenderTree
from utils.utils import load_pickle, save_as_pickle
from net.net import AlphaNet
from game.game import Game


class MCTSNode():
    def __init__(self, game, move, n_children, parent=None):
        self.game = game  # state s
        self.move = move  # action index
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.n_children = n_children
        self.child_priors = np.zeros([self.n_children], dtype=np.float32)
        self.child_total_value = np.zeros([self.n_children], dtype=np.float32)
        self.child_number_visits = np.zeros([self.n_children], dtype=np.float32)
        self.action_idxes = []

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return math.sqrt(self.number_visits) * (
                abs(self.child_priors) / (1 + self.child_number_visits))

    def best_child(self):
        if self.action_idxes != []:
            bestmove = self.child_Q() + self.child_U()
            bestmove = self.action_idxes[np.argmax(bestmove[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_Q() + self.child_U())
        return bestmove

    def select_leaf(self):
        current = self
        while current.is_expanded:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current

    def add_dirichlet_noise(self, action_idxs, child_priors):
        valid_child_priors = child_priors[action_idxs]  # select only legal moves entries in child_priors array
        valid_child_priors = 0.75 * valid_child_priors + 0.25 * np.random.dirichlet(np.zeros([len(valid_child_priors)], \
                                                                                             dtype=np.float32) + 192)
        child_priors[action_idxs] = valid_child_priors
        return child_priors

    def expand(self, child_priors):
        self.is_expanded = True
        action_idxs = self.game.actions()
        c_p = child_priors
        if action_idxs == []:
            self.is_expanded = False
        self.action_idxes = action_idxs
        c_p[[i for i in range(len(child_priors)) if i not in action_idxs]] = 0.000000000  # mask all illegal actions
        if self.parent.parent == None:  # add dirichlet noise to child_priors in root node
            c_p = self.add_dirichlet_noise(action_idxs, c_p)
        self.child_priors = c_p

    def decode_n_move_pieces(self, board, move):
        board.drop_piece(move)
        return board

    def maybe_add_child(self, move):
        if move not in self.children:
            copy_board = copy.deepcopy(self.game)  # make copy of board
            copy_board = self.decode_n_move_pieces(copy_board, move)
            self.children[move] = MCTSNode(copy_board, move, self.n_children, parent=self)
        return self.children[move]

    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            if current.game.player == 1:  # same as current.parent.game.player = 0
                current.total_value += (1 * value_estimate)  # value estimate +1 = O wins
            elif current.game.player == 0:  # same as current.parent.game.player = 1
                current.total_value += (-1 * value_estimate)
            current = current.parent


def decode_n_move_pieces(self, board, move):
    board.drop_piece(move)
    return board


def maybe_add_child(self, move):
    if move not in self.children:
        copy_board = copy.deepcopy(self.game)  # make copy of board
        copy_board = self.decode_n_move_pieces(copy_board, move)
        self.children[move] = MCTSNode(copy_board, move, n_children=self.n_children, parent=self)
    return self.children[move]


def backup(self, value_estimate: float):
    current = self
    while current.parent is not None:
        current.number_visits += 1
        if current.game.player == 1:  # same as current.parent.game.player = 0
            current.total_value += (1 * value_estimate)  # value estimate +1 = O wins
        elif current.game.player == 0:  # same as current.parent.game.player = 1
            current.total_value += (-1 * value_estimate)
        current = current.parent


class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
