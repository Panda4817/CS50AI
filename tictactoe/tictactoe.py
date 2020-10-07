"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    
    # Keep track of number of Xs
    Xnum = 0
    # Keep track of number of Os
    Onum = 0

    # Loop through board counting number of x and o
    for row in board:
        for cell in row:
            if cell == X:
                Xnum += 1
            if cell == O:
                Onum += 1
    
    # If x is more than o then it is O's turn
    if Xnum > Onum:
        return O
    # Else if O is more than X then it is X's turn
    elif Onum > Xnum:
        return X
    # Else it is X's turn as x goes first
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # Initialise a set for storing possible moves
    moves = set()

    # Loop through board finding the empty cells and storing it as (row, cell)
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                moves.add((i, j))

    return moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # Make a deep copy of the board so we don't mutate the original board
    copyboard = copy.deepcopy(board)

    # Check action is valid by checking the (row, cell) is empty
    if copyboard[action[0]][action[1]] == EMPTY:
        copyboard[action[0]][action[1]] = player(board)
    # Else raise an error
    else:
        raise ReferenceError('Invalid Action')

    return copyboard


def checkPosition(ls):
    """
    Returns true or false depeding on if a winning state is reached or not.
    """
    # This if statement checks if 3 in a row is achieved through middle cell of the board
    if (1, 1) in ls:
        if (0, 0) in ls and (2, 2) in ls:
            return True
        if (0, 2) in ls and (2, 0) in ls:
            return True
        if (1, 0) in ls and (1, 2) in ls:
            return True
        if (0, 1) in ls and (2, 1) in ls:
            return True
    
    # These if statemnets check if 3 in a row is acheived through the outer cells of the board
    if (0, 0) in ls and (0, 1) in ls and (0, 2) in ls:
        return True
    if (2, 0) in ls and (2, 1) in ls and (2, 2) in ls:
        return True
    if (0, 0) in ls and (1, 0) in ls and (2, 0) in ls:
        return True
    if (0, 2) in ls and (1, 2) in ls and (2, 2) in ls:
        return True
    
    # If none of the if statements above returned true, the board is notin a winning state so return false as no one has won (yet)
    return False


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # A list of (row, cell) pairs which contain X
    Xpos = []
    # A list of (row, cell) pairs which contain O
    Opos = []

    # Loop through board to find the (row, cell) pairs
    for i in range(3):
        for j in range(3):
            if board[i][j] == X:
                Xpos.append((i, j))
            if board[i][j] == O:
                Opos.append((i, j))

    # Check of number of (row, cell) pairs is at least 3
    if len(Xpos) >= 3:
        # Use custom function to check if Xs are in the winning positions
        if checkPosition(Xpos):
            return X
    
    if len(Opos) >= 3:
        # Use custom function to check if Os are in the winning positions
        if checkPosition(Opos):
            return O
    
    # Else function returns None as it is a tie or game is still in progress
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # Check if the board has a winner, if not, check for empty cells
    if winner(board) == None:
        for row in board:
            for cell in row:
                if cell == EMPTY:
                    # Empty cell indicates that at least one move is left to play
                    return False
        # No empty cells mean this is the terminal board
        return True
    # If winner then this is the terminal board
    else:
        return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    # Find out who has won from the board
    win = winner(board)
    
    # If it is X, return max point
    if win == X:
        return 1
    # If it is O, return min point
    elif win == O:
        return -1
    # Else return 0, indicating a tie
    else:
        return 0


def maxValue(board, alpha, beta):
    """
    Returns the best utility for an action with alpha beta pruning for a max-player.
    """
    # Check the board is not terminal, if so, return utility
    if terminal(board):
        return utility(board)
    # Initiate a variable to store bestv utility, which for a max player is the lowest value possible
    bestv = float("-inf")

    # Loop through actions, calculating the minimum value, uses recursion
    for action in actions(board):
        v = minValue(result(board, action), alpha, beta)
        # Find the max value between current v and bestv and update bestv
        bestv = max(bestv, v)
        # Find max between alpha and bestv and update alpha
        alpha = max(alpha, bestv)
        # Check if beta is less than or = to alpha, if so stop searching tree and return bestv
        if beta <= alpha:
            break
    return bestv


def minValue(board, alpha, beta):
    """
    Returns the best utility for an action with alpha beta pruning for a min-player.
    """
    # Check the board is not terminal, if so, return utility
    if terminal(board):
        return utility(board)
    # Initiate a variable to store bestv utility, which for a min player is the highest value possible
    bestv = float("inf")
    
    # Loop through actions, calculating the maximum value, uses recursion
    for action in actions(board):
        v = maxValue(result(board, action), alpha, beta)
        # Find the min value between current v and bestv and update bestv
        bestv = min(bestv, v)
        # Find min between beta and bestv and update beta
        beta = min(beta, bestv)
        # Check if beta is less than or = to alpha, if so stop searching tree and return bestv
        if beta <= alpha:
            break
    return bestv


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # Check if board is terminal, if so return None
    if terminal(board):
        return None
    
    # Workout twhose turn it is using the player function
    pl = player(board)

    # Workout all the possible moves from the board
    moves = actions(board)

    # Initiate a list, which will store {"moves": action, "v": best utility value}
    utility = []

    # Inititate a explored list to keep track of the actions already explored
    explored = set()

    # Initiate the best alpha and beta variables (alpha is for the max-player and beta is for the min-player)
    alpha = float("-inf")
    beta = float("inf")

    # If next player is x, use the minValue function
    if pl == X:
        # Loop through actions
        for move in moves:
            # Check the move is not in explored
            if not move in explored:
                # Add to list the (row, cell) pair as move and call minValue function, to find opponents best value and from that find your best value
                utility.append({'move': move, 'v': minValue(result(board, move), alpha, beta)})
                # Add the move to explored
                explored.add(move)
        # Find the max value from all actions
        optimal = max(utility, key=lambda x: x['v'])
        return optimal['move']
    
    # If the next player is O, use the max-value function
    if pl == O:
        # Loop through actions
        for move in moves:
            # Check the move is not in explored
            if not move in explored:
                # Add to list the (row, cell) pair as move and call maxValue function, to find opponents best value and from that find your best value
                utility.append({'move': move, 'v': maxValue(result(board, move), alpha, beta)})
                # Add the move to explored
                explored.add(move)
        # Find the min value from all actions 
        optimal = min(utility, key=lambda x: x['v'])
        return optimal['move']  

