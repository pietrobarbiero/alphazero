import unittest


class TestConnect4(unittest.TestCase):

    def test_connect4(self):

        from alphazero import learn, evaluate
        from alphazero import run_MCTS
        from alphazero import Connect4
        from alphazero import AlphaNet
        from argparse import ArgumentParser
        import logging

        logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
        logger = logging.getLogger(__file__)

        parser = ArgumentParser()
        parser.add_argument("--iteration", type=int, default=0, help="Current iteration number to resume from")
        parser.add_argument("--total_iterations", type=int, default=2, #default=1000,
                            help="Total number of iterations to run")
        parser.add_argument("--MCTS_num_processes", type=int, default=1, # 5
                            help="Number of processes to run MCTS self-plays")
        parser.add_argument("--num_games_per_MCTS_process", type=int, default=2, # 120
                            help="Number of games to simulate per MCTS self-play process")
        parser.add_argument("--temperature_MCTS", type=float, default=1.1,
                            help="Temperature for first 10 moves of each MCTS self-play")
        parser.add_argument("--num_evaluator_games", type=int, default=100,
                            help="No of games to play to evaluate neural nets")
        parser.add_argument("--neural_net_name", type=str, default="cc4_current_net_", help="Name of neural net")
        parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
        parser.add_argument("--num_epochs", type=int, default=30, # 300
                            help="No of epochs")
        parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
        parser.add_argument("--gradient_acc_steps", type=int, default=1,
                            help="Number of steps of gradient accumulation")
        parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
        args = parser.parse_args()

        logger.info("Starting iteration pipeline...")
        for i in range(args.iteration, args.total_iterations):
            run_MCTS(args, AlphaNet, Connect4, start_idx=0, iteration=i)
            learn(args, AlphaNet, Connect4, iteration=i, new_optim_state=True)
            if i >= 1:
                winner = evaluate(args, i, i + 1, AlphaNet, Connect4)
                counts = 0
                while (winner != (i + 1)):
                    logger.info("Trained net didn't perform better, generating more MCTS games for retraining...")
                    run_MCTS(args, AlphaNet, Connect4, start_idx=(counts + 1) * args.num_games_per_MCTS_process, iteration=i)
                    counts += 1
                    learn(args, AlphaNet, Connect4, iteration=i, new_optim_state=True)
                    winner = evaluate(args, i, i + 1, AlphaNet, Connect4)
        print()
    #
    # def test_mcts(self):
    #
    #     from alphazero import Board, view_board, ConnectNet
    #
    #     net = ConnectNet()
    #
    #     print()
    #
    #
    # def test_self_play(self):
    #
    #     from alphazero import Connect4
    #     from alphazero import AlphaNet
    #     from alphazero import self_play
    #     import torch
    #     import logging
    #     import os
    #
    #     num_games = 2   # number of games to simulate per MCTS self-play process
    #     start_idx = 0
    #     iteration = 0   # current iteration number to resume from
    #     temperature_mcts = 1.1  # temperature for first 10 moves of each MCTS self-play
    #     total_iterations = 10
    #     neural_net_name = "cc4_current_net_"
    #
    #     c4game = Connect4()
    #     net = AlphaNet(c4game)
    #     cuda = torch.cuda.is_available()
    #     if cuda:
    #         net.cuda()
    #
    #     model_data_dir = "./model_data/"
    #     if not os.path.isdir(model_data_dir):
    #         os.makedirs(model_data_dir)
    #
    #     logging.info("Starting iteration pipeline...")
    #     for i in range(iteration, total_iterations):
    #
    #         net_to_play = "%s_iter%d.pth.tar" % (neural_net_name, iteration)
    #
    #         logging.info("Preparing model for MCTS...")
    #         net.eval()
    #
    #         current_net_filename = os.path.join(model_data_dir, net_to_play)
    #         if os.path.isfile(current_net_filename):
    #             checkpoint = torch.load(current_net_filename)
    #             net.load_state_dict(checkpoint['state_dict'])
    #             logging.info("Loaded %s model." % current_net_filename)
    #         else:
    #             torch.save({'state_dict': net.state_dict()}, os.path.join(model_data_dir, net_to_play))
    #             logging.info("Initialized model.")
    #
    #         with torch.no_grad():
    #             cpu = 0
    #             self_play(net, c4game, num_games, start_idx, cpu, temperature_mcts, iteration)
    #         logging.info("Finished MCTS!")
    #
    # def test_game(self):
    #
    #     from alphazero import Connect4
    #     from alphazero import AlphaNet
    #     import torch
    #
    #     c4game = Connect4()
    #     fig = c4game.view_game()
    #     fig.show()
    #     c4game.move(5)
    #     fig = c4game.view_game()
    #     fig.show()
    #     winner = c4game.check_winner()
    #     actions = c4game.actions()
    #     print(winner)
    #     print(actions)
    #
    #     encoded = c4game.encode_state()
    #     decoded = c4game.decode_state(encoded)
    #     encoded_t = encoded.transpose(2, 0, 1)
    #     encoded_s = torch.from_numpy(encoded_t).float().cuda()
    #
    #     net = AlphaNet(c4game)
    #     cuda = torch.cuda.is_available()
    #     if cuda:
    #         net.cuda()
    #     net.eval()
    #
    #     child_priors, value_estimate = net(encoded_s)
    #     child_priors = child_priors.detach().cpu().numpy().reshape(-1)
    #     value_estimate = value_estimate.item()
    #
    #     print()


suite = unittest.TestLoader().loadTestsFromTestCase(TestConnect4)
unittest.TextTestRunner(verbosity=2).run(suite)
