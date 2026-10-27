"""
Interactive CLI to play Connect-4 against any trained Agent, Benchmark Opponent, or Solver.
Usage:
    python play_agent.py --checkpoint results/teachers/teacher_round3_improved.pt
    python play_agent.py --opponent heuristic
"""

import sys
import argparse
from pathlib import Path

from src.game.connect4 import Connect4Board, PLAYER_1, PLAYER_2
from src.eval.opponents import RandomPlayer, HeuristicPlayer, MinimaxPlayer, ModelPlayer
from src.model.train import load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Play Connect-4 against an AI Agent")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--opponent", type=str, default="heuristic", choices=["model", "heuristic", "minimax", "random"], help="Opponent type if no checkpoint")
    parser.add_argument("--first", action="store_true", help="Play as Player 1 (Red / X) moving first")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load agent
    if args.checkpoint and Path(args.checkpoint).exists():
        model, meta = load_checkpoint(args.checkpoint)
        agent = ModelPlayer(model, greedy=True, name=f"Neural Agent (R{meta.get('round_num', '?')})")
    elif args.opponent == "heuristic":
        agent = HeuristicPlayer("Heuristic AI")
    elif args.opponent == "minimax":
        agent = MinimaxPlayer(depth=4, name="Minimax AI (Depth 4)")
    elif args.opponent == "random":
        agent = RandomPlayer()
    else:
        agent = HeuristicPlayer("Default Heuristic AI")

    human_player = PLAYER_1 if args.first else PLAYER_2
    ai_player = -human_player

    board = Connect4Board()
    print("=" * 45)
    print("WELCOME TO CONNECT-4 CLI")
    print(f"You are playing as {'X (P1)' if human_player == 1 else 'O (P2)'}")
    print(f"Opponent: {agent.name}")
    print("=" * 45)

    while not board.is_terminal():
        print("\nCurrent Board:")
        print(board)
        legal = board.legal_moves()

        if board.current_player == human_player:
            valid_move = False
            while not valid_move:
                try:
                    user_input = input(f"Enter column (legal: {legal}) or 'q' to quit: ").strip()
                    if user_input.lower() == 'q':
                        print("Game aborted.")
                        return
                    col = int(user_input)
                    if col in legal:
                        board.make_move(col)
                        valid_move = True
                    else:
                        print(f"Column {col} is not legal. Try again.")
                except ValueError:
                    print("Invalid input. Please enter an integer between 0 and 6.")
        else:
            print(f"\n{agent.name} is thinking...")
            col = agent.get_move(board)
            print(f"{agent.name} played column {col}")
            board.make_move(col)

    print("\n" + "=" * 45)
    print("GAME OVER")
    print("=" * 45)
    print(board)
    winner = board.winner()
    if winner == human_player:
        print("CONGRATULATIONS! You won!")
    elif winner == ai_player:
        print(f"{agent.name} won!")
    else:
        print("It's a DRAW!")

if __name__ == "__main__":
    main()
