import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table
from ..game import Game


class Connect4(Game):

    def __init__(self):
        super().__init__()
        self.init_state = np.zeros([6, 7]).astype(str)
        self.init_state[self.init_state == "0.0"] = " "
        self.current_state = self.init_state
        self.player = 0

        # network parameters
        self.game_state_shape = [-1, 3, 6, 7]  # batch_size x channels x board_x x board_y
        self.n_conv_blocks = 1
        self.n_res_blocks = 10  # 19
        self.game_dim = 6 * 7
        self.action_size = 7

    def encode_state(self):
        board_state = self.current_state
        encoded = np.zeros([6, 7, 3]).astype(int)
        encoder_dict = {"O": 0, "X": 1}
        for row in range(6):
            for col in range(7):
                if board_state[row, col] != " ":
                    encoded[row, col, encoder_dict[board_state[row, col]]] = 1
        if self.player == 1:
            encoded[:, :, 2] = 1  # player to move
        return encoded

    def decode_state(self, encoded):
        decoded = np.zeros([6, 7]).astype(str)
        decoded[decoded == "0.0"] = " "
        decoder_dict = {0: "O", 1: "X"}
        for row in range(6):
            for col in range(7):
                for k in range(2):
                    if encoded[row, col, k] == 1:
                        decoded[row, col] = decoder_dict[k]
        cboard = Connect4()
        cboard.current_state = decoded
        cboard.player = encoded[0, 0, 2]
        return cboard

    def move(self, column):
        if self.current_state[0, column] != " ":
            return "Invalid move"
        else:
            row = 0
            pos = " "
            while (pos == " "):
                if row == 6:
                    row += 1
                    break
                pos = self.current_state[row, column]
                row += 1
            if self.player == 0:
                self.current_state[row - 2, column] = "O"
                self.player = 1
            elif self.player == 1:
                self.current_state[row - 2, column] = "X"
                self.player = 0

    def move_back(self, column):
        if self.current_state[0, column] != " ":
            return "Invalid move"
        else:
            row = 0
            pos = " "
            while pos == " ":
                if row == 6:
                    row += 1
                    break
                pos = self.current_state[row, column]
                row += 1
            if self.player == 0:
                self.current_state[row - 2, column] = "O"
                self.player = 1
            elif self.player == 1:
                self.current_state[row - 2, column] = "X"
                self.player = 0

    def check_winner(self):
        if self.player == 1:
            for row in range(6):
                for col in range(7):
                    if self.current_state[row, col] != " ":
                        # rows
                        try:
                            if self.current_state[row, col] == "O" and self.current_state[row + 1, col] == "O" and \
                                    self.current_state[row + 2, col] == "O" and self.current_state[row + 3, col] == "O":
                                # print("row")
                                return True
                        except IndexError:
                            next
                        # columns
                        try:
                            if self.current_state[row, col] == "O" and self.current_state[row, col + 1] == "O" and \
                                    self.current_state[row, col + 2] == "O" and self.current_state[
                                row, col + 3] == "O":
                                # print("col")
                                return True
                        except IndexError:
                            next
                        # \ diagonal
                        try:
                            if self.current_state[row, col] == "O" and self.current_state[
                                row + 1, col + 1] == "O" and \
                                    self.current_state[row + 2, col + 2] == "O" and self.current_state[
                                row + 3, col + 3] == "O":
                                # print("\\")
                                return True
                        except IndexError:
                            next
                        # / diagonal
                        try:
                            if self.current_state[row, col] == "O" and self.current_state[
                                row + 1, col - 1] == "O" and \
                                    self.current_state[row + 2, col - 2] == "O" and self.current_state[
                                row + 3, col - 3] == "O" \
                                    and (col - 3) >= 0:
                                # print("/")
                                return True
                        except IndexError:
                            next
        if self.player == 0:
            for row in range(6):
                for col in range(7):
                    if self.current_state[row, col] != " ":
                        # rows
                        try:
                            if self.current_state[row, col] == "X" and self.current_state[row + 1, col] == "X" and \
                                    self.current_state[row + 2, col] == "X" and self.current_state[
                                row + 3, col] == "X":
                                return True
                        except IndexError:
                            next
                        # columns
                        try:
                            if self.current_state[row, col] == "X" and self.current_state[row, col + 1] == "X" and \
                                    self.current_state[row, col + 2] == "X" and self.current_state[
                                row, col + 3] == "X":
                                return True
                        except IndexError:
                            next
                        # \ diagonal
                        try:
                            if self.current_state[row, col] == "X" and self.current_state[
                                row + 1, col + 1] == "X" and \
                                    self.current_state[row + 2, col + 2] == "X" and self.current_state[
                                row + 3, col + 3] == "X":
                                return True
                        except IndexError:
                            next
                        # / diagonal
                        try:
                            if self.current_state[row, col] == "X" and self.current_state[
                                row + 1, col - 1] == "X" and \
                                    self.current_state[row + 2, col - 2] == "X" and self.current_state[
                                row + 3, col - 3] == "X" \
                                    and (col - 3) >= 0:
                                return True
                        except IndexError:
                            next

    def actions(self):  # returns all possible moves
        acts = []
        for col in range(7):
            if self.current_state[0, col] == " ":
                acts.append(col)
        return acts

    def view_game(self, initial=False, fmt='{:s}', bkg_colors=['pink', 'pink']):
        if initial:
            np_data = self.init_state
        else:
            np_data = self.current_state
        data = pd.DataFrame(np_data, columns=['0', '1', '2', '3', '4', '5', '6'])
        fig, ax = plt.subplots(figsize=[7, 7])
        ax.set_axis_off()
        tb = Table(ax, bbox=[0, 0, 1, 1])
        nrows, ncols = data.shape
        width, height = 1.0 / ncols, 1.0 / nrows

        for (i, j), val in np.ndenumerate(data):
            idx = [j % 2, (j + 1) % 2][i % 2]
            color = bkg_colors[idx]

            tb.add_cell(i, j, width, height, text=fmt.format(val),
                        loc='center', facecolor=color)

        for i, label in enumerate(data.index):
            tb.add_cell(i, -1, width, height, text=label, loc='right',
                        edgecolor='none', facecolor='none')

        for j, label in enumerate(data.columns):
            tb.add_cell(-1, j, width, height / 2, text=label, loc='center',
                        edgecolor='none', facecolor='none')
        tb.set_fontsize(24)
        ax.add_table(tb)
        # ax.margins(20)
        return fig
