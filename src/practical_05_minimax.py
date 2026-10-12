"""Practical 5 — Minimax Territory Control Game (5x5 board).

AI is player X, opponent is O.  Players start in opposite corners and claim
an empty cell adjacent to a cell they already control.  A player with no legal
move passes; the game ends when neither can move.  Most cells controlled wins.
Minimax depth limit: 4 plies.
"""

from __future__ import annotations

import copy

N = 5
X, O, EMPTY = "X", "O", "."
DEPTH_LIMIT = 4
START_BOARD = [[EMPTY] * N for _ in range(N)]
START_BOARD[0][0] = X
START_BOARD[N - 1][N - 1] = O
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def legal_moves(board, player):
    """Return list of empty cells adjacent to a cell controlled by player."""
    moves = []
    for r in range(N):
        for c in range(N):
            if board[r][c] != EMPTY:
                continue
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N and board[nr][nc] == player:
                    moves.append((r, c))
                    break
    return moves


def apply_move(board, move, player):
    nb = copy.deepcopy(board)
    nb[move[0]][move[1]] = player
    return nb


def evaluate(board, player):
    """Territory difference + frontier (adjacent empty) opportunities."""
    opp = O if player == X else X
    mine = sum(row.count(player) for row in board)
    theirs = sum(row.count(opp) for row in board)
    frontier = 0
    for r in range(N):
        for c in range(N):
            if board[r][c] != EMPTY:
                continue
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N and board[nr][nc] == player:
                    frontier += 1
                    break
    return (mine - theirs) + 0.1 * frontier


def is_terminal(board):
    return not legal_moves(board, X) and not legal_moves(board, O)


def minimax(board, depth, maximizing, ai_player):
    opp = O if ai_player == X else X
    current = ai_player if maximizing else opp
    if depth == 0 or is_terminal(board):
        return evaluate(board, ai_player), None
    moves = legal_moves(board, current)
    if not moves:
        return minimax(board, depth - 1, not maximizing, ai_player)
    best_move = None
    if maximizing:
        best_val = float("-inf")
        for m in moves:
            val, _ = minimax(apply_move(board, m, current), depth - 1, False, ai_player)
            if val > best_val:
                best_val, best_move = val, m
        return best_val, best_move
    else:
        best_val = float("inf")
        for m in moves:
            val, _ = minimax(apply_move(board, m, current), depth - 1, True, ai_player)
            if val < best_val:
                best_val, best_move = val, m
        return best_val, best_move


def choose_ai_move(board):
    """Pick the AI's (X) move using depth-limited Minimax."""
    _, move = minimax(board, DEPTH_LIMIT, True, X)
    if move is None:
        moves = legal_moves(board, X)
        move = moves[0] if moves else None
    return move


def opponent_policy(board):
    """Greedy opponent: claim the move with most empty neighbours (frontier)."""
    moves = legal_moves(board, O)
    if not moves:
        return None
    def score(m):
        r, c = m
        return sum(
            1 for dr, dc in DIRECTIONS
            if 0 <= r + dr < N and 0 <= c + dc < N and board[r + dr][c + dc] == EMPTY
        )
    return max(moves, key=score)


def render(board):
    return "\n".join(" ".join(row) for row in board)


def run(verbose=True):
    """Play a full AI-vs-greedy game and return the final board."""
    board = copy.deepcopy(START_BOARD)
    turn = X
    passes = 0
    if verbose:
        print("Initial board:")
        print(render(board))
    while passes < 2:
        if turn == X:
            move = choose_ai_move(board)
        else:
            move = opponent_policy(board)
        if move is None:
            passes += 1
            if verbose:
                print(f"\n{turn} passes (no legal move).")
        else:
            board = apply_move(board, move, turn)
            passes = 0
            if verbose:
                print(f"\n{turn} claims {move}:")
                print(render(board))
        turn = O if turn == X else X
    x = sum(row.count(X) for row in board)
    o = sum(row.count(O) for row in board)
    if verbose:
        print(f"\nFinal score -> X (AI): {x}, O (opponent): {o}")
        print("Winner:", "X (AI)" if x > o else "O (opponent)" if o > x else "Draw")
    return board


if __name__ == "__main__":
    run()
