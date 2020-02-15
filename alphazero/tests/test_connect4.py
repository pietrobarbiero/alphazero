import unittest


class TestConnect4(unittest.TestCase):

    def test_connect4(self):

        from alphazero import learn, evaluate
        from alphazero import run_MCTS
        from alphazero import Connect4
        from alphazero import AlphaNet
        from argparse import ArgumentParser
        import logging

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
        parser.add_argument("--num_evaluator_games", type=int, default=1,
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

        logging.info("Starting iteration pipeline...")
        for i in range(args.iteration, args.total_iterations):
            run_MCTS(args, AlphaNet, Connect4, start_idx=0, iteration=i)
            learn(args, AlphaNet, Connect4, iteration=i, new_optim_state=True)
            if i >= 1:
                winner = evaluate(args, i, i + 1, AlphaNet, Connect4)
                counts = 0
                while (winner != (i + 1)):
                    logging.info("Trained net didn't perform better, generating more MCTS games for retraining...")
                    run_MCTS(args, AlphaNet, Connect4, start_idx=(counts + 1) * args.num_games_per_MCTS_process, iteration=i)
                    counts += 1
                    learn(args, AlphaNet, Connect4, iteration=i, new_optim_state=True)
                    winner = evaluate(args, i, i + 1, AlphaNet, Connect4)
        print()


suite = unittest.TestLoader().loadTestsFromTestCase(TestConnect4)
unittest.TextTestRunner(verbosity=2).run(suite)
