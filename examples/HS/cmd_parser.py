import os
import json
import re
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd


def _parse_card(card: str) -> Dict:
    """
    Parse card.

    :param card: card info as string
    :return: card info as dictionary
    """
    re_result = re.findall(r"\[.*?\]", card)

    # get position
    p = re_result[0].replace("]", "").split("P")[1]
    position = json.loads(p)

    try:
        # get attack/health/durability
        a, h = re_result[1].replace("[", "").replace("]", "").split("/")
        attack, health = json.loads(a), json.loads(h)
        cost_string = re_result[2]
        id_string = re_result[3]

    except ValueError:
        attack, health = None, None
        cost_string = re_result[1]
        id_string = re_result[2]

    # get cost
    c = cost_string.replace("]", "").split("C")[1]
    cost = json.loads(c)

    # get ID
    card_id = json.loads(id_string)[0]

    # get name
    name = re.findall(r"\]'(.*)\[", card)[0]

    card = {
        "ID": card_id,
        "name": name,
        "position": position,
        "attack": attack,
        "health": health,
        "cost": cost
    }

    return card


def _parse_zone_info(zone_info: str) -> Dict:
    """
    Parse zone info.

    Notes
    -----
    A zone can be:
    - the player hand
    - the board
    - the deck

    :param zone_info: zone info as string
    :return: zone info as dictionary
    """
    card_list = zone_info.split("|")[:-1]

    # get player ID
    player_info = card_list.pop(0)
    player_id_str = player_info.replace("']", "").split("Player")[1]
    player_id = json.loads(player_id_str)

    # parse cards
    cards = []
    for card in card_list:
        cards.append(_parse_card(card))

    zone = {
        "playerID": player_id,
        "cards": cards,
    }

    return zone


def _parse_hero_info(hero_info: str) -> Dict:
    """
    Parse hero info.

    :param hero_info: hero info as string
    :return: hero info as dictionary
    """
    re_result = re.findall(r"\[.*?\]", hero_info)

    # get name
    hero_name_and_id = re_result[1].split("'")[1]
    hero_name = hero_name_and_id.split("[")[0]

    # get ID
    hero_id_str = re.findall(r"\[([A-Za-z0-9_]+)\]", hero_name_and_id)[0]
    hero_id = json.loads(hero_id_str)

    # get details
    details = re_result[2].replace("[", "").replace("]", "").split("/")
    attack = json.loads(details[0].split("ATK")[1])
    armor = json.loads(details[1].split("AR")[1])
    hp = json.loads(details[2].split("HP")[1])

    # get weapon
    weapon = re_result[3].replace("[", "").replace("]", "")

    # get spell power
    sp = re_result[4].replace("[", "").replace("]", "")
    spell_power_id = json.loads(sp.split("SP")[1])

    hero_info = {
        "ID": hero_id,
        "name": hero_name,
        "attack": attack,
        "armor": armor,
        "hp": hp,
        "weapon": weapon,
        "spell_power_id": spell_power_id
    }

    return hero_info


def _parse_game_state(game_state: str) -> Dict:
    """
    Parse game state.

    :param game_state: game state representation as string
    :return: game state representation as dictionary
    """
    item_list = game_state.split("\n")

    # get turn ID
    turn_id = json.loads(item_list[0])
    # get player ID
    re_result = re.findall(r"\[([A-Za-z0-9_]+)\]", item_list[1])
    player_id = json.loads(re_result[0])
    # get hero info
    hero_info = _parse_hero_info(item_list[2])
    # get hand info
    hand_info = _parse_zone_info(item_list[3])
    # get board info
    board_info = _parse_zone_info(item_list[4])
    # get deck info
    deck_info = _parse_zone_info(item_list[5])
    # get number of cards in opponent's hand
    opponent_n_cards = json.loads(item_list[6].split(":")[-1])

    game_state = {
        "turnID": turn_id,
        "playerID": player_id,
        "hero": hero_info,
        "hand": hand_info,
        "board": board_info,
        "deck": deck_info,
        "opp_hand": opponent_n_cards,
    }

    return game_state


def _parse_turn(turn: str) -> Dict:
    """
    Parse turn.

    :param turn: turn info
    :return: game state representation of the turn
    """
    task_list = turn.split("---->POSSIBLE TASK:")
    gs = task_list.pop(0)

    game_state = _parse_game_state(gs)

    return game_state


def _parse(game: str) -> List:
    """
    Parse HeartStone game using SabberStone output.

    :param game: detailed command line output of the game
    :return: list of game states
    """
    turn_list = game.split("--- TURN:")
    turn_list.pop(0)

    state_list = []
    for turn in turn_list:
        state_list.append(_parse_turn(turn))

    return state_list


def state_to_array(state: Dict) -> np.array:
    """
    Transform state dictionary to numpy array

    :param state: state representation as dictionary
    :return: state representation as numpy array
    """
    # check hero weapon!
    hero = pd.DataFrame.from_dict(state["hero"], orient='index').T
    hero.drop(["name", "weapon"], axis=1, inplace=True)

    # transform zones to dataframe
    columns = ["ID", "position",
               "attack", "health", "cost"]
    zone_list = [
        state["hand"]["cards"],
        state["board"]["cards"],
        state["deck"]["cards"],
    ]
    # max cards allowed per: hand, board, deck
    size_list = [10, 7, 30]

    # merge cards from ALL zones
    all_cards = pd.DataFrame()
    for cards, size in zip(zone_list, size_list):
        df_cards = pd.DataFrame(cards)
        if not df_cards.empty:
            df_cards.drop("name", axis=1, inplace=True)
            df_cards.sort_values(by=["position"])
            size -= len(df_cards)

        df_zero = pd.DataFrame(0, index=range(size), columns=columns)
        df_cards = df_cards.append(df_zero)
        all_cards = all_cards.append(df_cards, ignore_index=True)

    all_cards.fillna(0, inplace=True)

    # build state array
    hero_array = hero.values.flatten().astype(np.float)
    card_array = all_cards.values.flatten().astype(np.float)
    state_array = np.concatenate([hero_array, card_array])

    return state_array


def _parse_game(game: str) -> np.array:
    """
    Parse SabberStone game

    :param game: detailed command line output of the game
    :return: game dataset
    """
    # parse game and get a list of states
    state_list = _parse(game)

    # build the dataset
    dataset = np.array([], dtype=np.int64).reshape(0, 240)
    for state in state_list:
        state_array = state_to_array(state)
        if dataset.shape[0] == 0:
            dataset = state_array[:, np.newaxis].T

        else:
            state_array = state_array[:, np.newaxis].T
            dataset = np.append(dataset, state_array, axis=0)

    # the label of each state is: +1 for the winner, 0 for the loser
    labels = dataset[:, 0].astype(np.int)
    labels = (labels == labels[-1]).astype(np.int)
    labels = labels[:, np.newaxis]

    dataset = np.hstack([dataset, labels])

    return dataset


def _collapse_states(data: np.array) -> np.array:
    """
    Collapse dataset duplicates (duplicated states)

    :param data: data with duplicates
    :return: data without duplicates
    """
    df = pd.DataFrame(data)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # group duplicated states
    is_dup_list = X.duplicated(keep=False).values
    duplicate_indices = np.where(is_dup_list)[0]
    dup_state_groups = X.iloc[duplicate_indices].groupby(X.columns.tolist()).indices

    # collapse duplicates
    collapsed_states = X.duplicated(keep="first").values
    X_dup = X.iloc[collapsed_states]
    y_dup = np.zeros((sum(collapsed_states), 1))
    for i, (k, v) in enumerate(dup_state_groups.items()):
        y_old = y[duplicate_indices[v]]
        y_dup[i] = np.sum(y_old)
    dups = np.hstack([X_dup, y_dup])

    # get unique states (rows)
    X_unique = X.drop_duplicates(keep=False).values
    y_unique = y.iloc[~is_dup_list].values[:, np.newaxis]
    uniques = np.hstack([X_unique, y_unique])

    collapsed_data = np.vstack([dups, uniques])

    return collapsed_data


def parse_sabberstone_games(file_path: str) -> np.array:
    """
    Parse SabberStone games and get a dataset

    :param file_path: path to file containing detailed game information
    :return: dataset
    """
    # read file
    with open(file_path, "r") as f:
        cmd_output = f.read()

    # get list of games
    game_list = cmd_output.split("----- NEW GAME STARTING! -----")

    # build the dataset
    data = np.array([], dtype=np.int64).reshape(0, 241)
    for game in game_list[1:]:
        game_data = _parse_game(game)
        data = np.append(data, game_data, axis=0)

    # remove duplicated states
    data = _collapse_states(data)

    return data


if __name__ == '__main__':

    # os.getcwd()
    try:
        os.chdir("./examples/HS")
    except:
        pass
    file_path = "cmd_out.txt"
    file_out = "parsed_data.csv"

    print("Output will be dumped to \"%s\"..." % file_out)
    data = parse_sabberstone_games(file_path)

    np.savetxt(file_out, data, delimiter=",")

