#!/usr/bin/python3

from game import SnakeGame
from DQN import Model, DQN
import argparse
import sys


def parse_arg() -> argparse.Namespace:
    """Parse arguments and flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", "-i", type=int, default=1000000)
    parser.add_argument("--train", "-t", type=str, default="false")
    parser.add_argument("--visualize", "-v", type=str, default="true")
    parser.add_argument("--loadweights", "-l", type=str)
    parser.add_argument("--saveweights", "-s", type=str, default="params.json")
    args = parser.parse_args()
    return args


def check_args(args) -> tuple[bool]:
    """Check if arguments are correct, get the mode for program."""
    possible = ["true", "false"]
    if args.train not in possible or args.visualize not in possible:
        raise TypeError("Wrong argument type")
    if args.iteration < 1:
        raise TypeError("Wrong iteration value")
    train_mode = True if args.train == "true" else False
    visual_mode = True if args.visualize == "true" else False
    return train_mode, visual_mode


def main():
    dqn = None
    try:
        args = parse_arg()
        train_mode, visual_mode = check_args(args)
        model = Model.create_model()
        game = SnakeGame(render=visual_mode)
        dqn = DQN(game, model, weights=args.loadweights, train_mode=train_mode)
        dqn.run(max_iter=args.iteration, save=args.saveweights)
    except Exception as e:
        if dqn and args and args.train:
            dqn.model.save_weights_to(args.saveweights)
            dqn.model.plt.close('all')
        print("Error:", e)
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
