"""
Tournament match runner and game execution between any two players.
Alternates first player and records win/draw/loss rates deterministically.
"""

import random
from typing import Dict, Any, Optional, List
from src.game.connect4 import Connect4Board, PLAYER_1, PLAYER_2
from src.eval.opponents import Player

def play_game(player1: Player, player2: Player, verbose: bool = False) -> int:
    """
    Plays a single full Connect-4 game between player1 (P1, 1) and player2 (P2, -1).
    Returns:
        1: player1 won
       -1: player2 won
        0: draw
    """
    board = Connect4Board()
    while not board.is_terminal():
        current_actor = player1 if board.current_player == PLAYER_1 else player2
        move = current_actor.get_move(board)
        board.make_move(move)

    winner = board.winner()
    if verbose:
        print(f"Game finished. Winner: {winner}")
        print(board)
    return int(winner if winner is not None else 0)

def run_matchup(
    player_a: Player,
    player_b: Player,
    num_games: int = 20,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates player_a against player_b over num_games.
    Alternates who plays first:
    - In even games, player_a is Player 1 (plays first).
    - In odd games, player_b is Player 1 (plays first).
    Returns win/draw/loss counts and rates from player_a's perspective.
    """
    random.seed(seed)
    wins = 0
    draws = 0
    losses = 0

    for i in range(num_games):
        if i % 2 == 0:
            outcome = play_game(player_a, player_b)
            if outcome == 1:
                wins += 1
            elif outcome == -1:
                losses += 1
            else:
                draws += 1
        else:
            outcome = play_game(player_b, player_a)
            if outcome == 1:
                losses += 1
            elif outcome == -1:
                wins += 1
            else:
                draws += 1

    return {
        "opponent": player_b.name,
        "games_played": num_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": float(wins / num_games),
        "draw_rate": float(draws / num_games),
        "loss_rate": float(losses / num_games),
        "score": float((wins + 0.5 * draws) / num_games) # Chess/tournament style score
    }

def run_tournament(
    agent: Player,
    opponents: List[Player],
    num_games_per_opponent: int = 20,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates agent against a suite of opponents.
    """
    matchups = {}
    total_wins = 0
    total_draws = 0
    total_losses = 0
    total_games = 0

    for opp in opponents:
        res = run_matchup(agent, opp, num_games=num_games_per_opponent, seed=seed)
        matchups[opp.name] = res
        total_wins += res["wins"]
        total_draws += res["draws"]
        total_losses += res["losses"]
        total_games += res["games_played"]

    return {
        "matchups": matchups,
        "total_games": total_games,
        "total_wins": total_wins,
        "total_draws": total_draws,
        "total_losses": total_losses,
        "overall_win_rate": float(total_wins / max(1, total_games)),
        "overall_score": float((total_wins + 0.5 * total_draws) / max(1, total_games))
    }
