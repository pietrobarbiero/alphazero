import logging
import os
from tqdm import tqdm
import copy
import numpy as np
import datetime
from ..utils.utils import load_pickle, save_as_pickle
from ..game.game import Game
from .search import mcts
import torch
import torch.multiprocessing as mp



def run_MCTS(args, net_class, game_class, start_idx=0, iteration=0):
    net_to_play="%s_iter%d.pth.tar" % (args.neural_net_name, iteration)
    cuda = torch.cuda.is_available()
    game = game_class()
    net = net_class(game)
    if cuda:
        net.cuda()

    model_data_dir = "./model_data/"
    if not os.path.isdir(model_data_dir):
        os.makedirs(model_data_dir)

    if args.MCTS_num_processes > 1:
        logging.info("Preparing model for multi-process MCTS...")
        mp.set_start_method("spawn", force=True)
        net.share_memory()
        net.eval()

        current_net_filename = os.path.join(model_data_dir, net_to_play)
        if os.path.isfile(current_net_filename):
            checkpoint = torch.load(current_net_filename)
            net.load_state_dict(checkpoint['state_dict'])
            logging.info("Loaded %s model." % current_net_filename)
        else:
            torch.save({'state_dict': net.state_dict()}, os.path.join(model_data_dir, net_to_play))
            logging.info("Initialized model.")

        processes = []
        if args.MCTS_num_processes > mp.cpu_count():
            num_processes = mp.cpu_count()
            logging.info(
                "Required number of processes exceed number of CPUs! Setting MCTS_num_processes to %d" % num_processes)
        else:
            num_processes = args.MCTS_num_processes

        logging.info("Spawning %d processes..." % num_processes)
        with torch.no_grad():
            for i in range(num_processes):
                p = mp.Process(target=self_play,
                               args=(net, game_class, args.num_games_per_MCTS_process, start_idx, i, args.temperature_MCTS, iteration))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        logging.info("Finished multi-process MCTS!")

    elif args.MCTS_num_processes == 1:
        logging.info("Starting iteration pipeline...")
        logging.info("Preparing model for MCTS...")
        net.eval()

        current_net_filename = os.path.join(model_data_dir, net_to_play)
        if os.path.isfile(current_net_filename):
            checkpoint = torch.load(current_net_filename)
            net.load_state_dict(checkpoint['state_dict'])
            logging.info("Loaded %s model." % current_net_filename)
        else:
            torch.save({'state_dict': net.state_dict()}, os.path.join(model_data_dir, net_to_play))
            logging.info("Initialized model.")

        with torch.no_grad():
            cpu = 0
            self_play(net, game_class, args.num_games_per_MCTS_process, start_idx, cpu, args.temperature_MCTS, iteration)
        logging.info("Finished MCTS!")


def self_play(net, game_class, num_games, start_idx, cpu, temperature_mcts, iteration):
    logging.info("[CPU: %d]: Starting MCTS self-play..." % cpu)

    if not os.path.isdir("./datasets/iter_%d" % iteration):
        if not os.path.isdir("datasets"):
            os.mkdir("datasets")
        os.mkdir("datasets/iter_%d" % iteration)

    for idxx in tqdm(range(start_idx, num_games + start_idx)):
        logging.info("[CPU: %d]: Game %d" % (cpu, idxx))
        game = game_class()
        checkmate = False
        dataset = []  # to get state, policy, value for neural network training
        states = []
        value = 0
        move_count = 0
        while checkmate == False and game.actions() != []:
            if move_count < 11:
                t = temperature_mcts
            else:
                t = 0.1
            states.append(copy.deepcopy(game.current_state))
            board_state = copy.deepcopy(game.encode_state())
            root = mcts(game, 777, net, t, 7)
            policy = get_policy(root, t)
            print("[CPU: %d]: Game %d POLICY:\n " % (cpu, idxx), policy)
            chosen_move = np.random.choice(np.arange(game.action_size), p=policy)
            game = do_decode_n_move_pieces(game, chosen_move)  # decode move and move piece(s)
            dataset.append([board_state, policy])
            print("[Iteration: %d CPU: %d]: Game %d CURRENT BOARD:\n" % (iteration, cpu, idxx),
                  game.current_state, game.player)
            print(" ")
            if game.check_winner() is True:  # if somebody won
                if game.player == 0:  # black wins
                    value = -1
                elif game.player == 1:  # white wins
                    value = 1
                checkmate = True
            move_count += 1
        dataset_p = []
        for idx, data in enumerate(dataset):
            s, p = data
            if idx == 0:
                dataset_p.append([s, p, 0])
            else:
                dataset_p.append([s, p, value])
        del dataset
        save_as_pickle("iter_%d/" % iteration + \
                       "dataset_iter%d_cpu%i_%i_%s" % (
                           iteration, cpu, idxx, datetime.datetime.today().strftime("%Y-%m-%d")), dataset_p)


def do_decode_n_move_pieces(board: Game, chosen_move):
    board.move(chosen_move)
    return board


def get_policy(root, temp=1):
    # policy = np.zeros([7], dtype=np.float32)
    # for idx in np.where(root.child_number_visits!=0)[0]:
    #    policy[idx] = ((root.child_number_visits[idx])**(1/temp))/sum(root.child_number_visits**(1/temp))
    result = ((root.child_number_visits) ** (1 / temp)) / sum(root.child_number_visits ** (1 / temp))
    if np.any(np.isnan(result)):
        print()
    return result
