#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import random
import math
from collections import deque
from collections import Counter
import json
import timeit
from copy import deepcopy


WIN_REWARD = 1.0
LOSS_REWARD = -1.0
TIE_REWARD = 0 

# future improvement
# heuristic functions on this paper: http://erikvanderwerf.tengen.nl/pubdown/thesis_erikvanderwerf.pdf

class Qlearner():
    def __init__(self, alpha=0.7, gamma=0.9, initial_value=0, Qfile='Qvalues.txt', save_Q=True, best_move=False):
        
        if not (0 < gamma <= 1):
            raise ValueError("")

        self.type = 'Qlearner'
        self.alpha = alpha
        self.gamma = gamma
        if os.path.exists(Qfile):
            self.q_table = self.read_Q()
        else: 
            self.q_table = {'1':{}, '2':{}} # key is 25 digit number of 3 digits, black 1, white 2 and empty 0
        self.history_states = deque() 
        self.opponent_history_states = deque()
        self.initial_value = initial_value
        self.piece_type = None
        self.save_Q = save_Q
        self.best_move = best_move
        #self.n_move = 0

    def get_input(self, go, piece_type): 
        # A function that returns a move/an action
        # input go: Go instance.
        # input piece_type: 1('X') or 2('O').
        # output: (row, column) coordinate of input.

        self.piece_type = piece_type
        
        encoded = self.encode_state(go.board)
        
        if self.save_Q:
            if encoded != '0' * 25:
                self.Q_lookup(3-piece_type, go.board)
                self.opponent_history_states.append(encoded)
        
        stones = self._count_stones(encoded)
            
            
        possible_placements = []

        if stones == 0:
            possible_placements.append((2,2))


        elif stones == 1 and go.board[2][2] == 0:
            possible_placements.append((2,2))

        elif stones <= 6: 
            starting_points = self._starting_placements()
            for position in starting_points:
                i, j = position

                if go.valid_place_check(i, j, piece_type, test_check = True):

                    possible_placements.append((i, j))

        else: 
            for i in range(go.size):
                for j in range(go.size):
                    if go.valid_place_check(i, j, piece_type, test_check = True):
                        possible_placements.append((i,j))

        if not possible_placements:
            return "PASS"
        else:
            action = self._select_action(go.board, possible_placements)

        return action

    
    def _count_stones(self, board_state):
        return 25 - len(list(filter(lambda x: x == '0', board_state)))

    def _starting_placements(self):
        return [(1, 1), (1, 2), (1,3), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3)]

    def _select_action(self, board, possible_actions, epsilon=0.7):
        # Use epsilon-greedy algorithm to select a move
        
        q_vals = []

        for action in possible_actions:
            i, j = action
            board[i][j] = self.piece_type # make a hypothetical move on board
            #board_state = self.encode_state(board) # encode state 
            #look up q value of this state
            q_val = self.Q_lookup(self.piece_type, board)
            #q_val = self.Q_lookup(board_state)[0]
            q_vals.append(q_val)
            board[i][j] = 0

        if not self.best_move:
            
            prob = random.uniform(0,1)
            
            if prob < epsilon:
                move = random.choice(possible_actions)
                board[move[0]][move[1]] = self.piece_type 
                self.history_states.append(self.encode_state(board))
                board[move[0]][move[1]] = 0
                return move 

        max_indices = self._find_max(q_vals)
        max_index = random.choice(max_indices)

        move = possible_actions[max_index]
        board[move[0]][move[1]] = self.piece_type
        self.history_states.append(board)
        board[move[0]][move[1]] = 0
        
        return move
    
    def _find_max(self, qval_list):
        
        max_q_index = []

        max_val = -np.inf

        for i, val in enumerate(qval_list):
            if val > max_val:
                max_val = val
                max_q_index = [i]

            elif val == max_val:
                max_q_index.append(i)

        return max_q_index

    def encode_state(self, board):
        return ''.join([str(board[i][j]) for i in range(len(board)) for j in range(len(board))])

    def _check_reflection(self, piece_type, board):
        # rotate the board and check reflections of a given board state
        
        rotated_state = self.encode_state(np.array(board).T)
        if rotated_state in self.q_table[str(piece_type)]:
            return self.q_table[str(piece_type)][rotated_state][0]
        
        rotated_state = self.encode_state(np.array(board)[::-1, ::-1])
        if rotated_state in self.q_table[str(piece_type)]:
            return self.q_table[str(piece_type)][rotated_state][0]

        rotated_state = self.encode_state(np.array(board).T[::-1, ::-1])
        if rotated_state in self.q_table[str(piece_type)]:
            return self.q_table[str(piece_type)][rotated_state][0]

        return None

    def Q_lookup(self, piece_type, board):
        # intialize and look up Q values
        
        state = self.encode_state(board)

        if state not in self.q_table[str(piece_type)]:
            qval = self._check_reflection(str(piece_type), board)
            if qval is not None:
                if self.save_Q:
                    self.q_table[str(piece_type)][state] = [qval, -2.0]
                return qval
            else:
                # initial Q val and max Q among possbile actions in the next step
                if self.save_Q:
                    self.q_table[str(piece_type)][state] = [self.initial_value, -2.0]
                else:
                    return self.initial_value
        
        return self.q_table[str(piece_type)][state][0]


    def update_Q(self, piece_type, history_states, reward):

        tmp_max_qval = -np.inf # records the max_q_value of known possible next steps
        
        for i in range(len(history_states)):

            state = history_states.pop()

            qval, max_q_val = self.q_table[str(piece_type)][state]

            if tmp_max_qval < -1.0:
                self.q_table[str(piece_type)][state][0] = reward
            
            else:    
                if tmp_max_qval > max_q_val:
                    
                    max_q_val = tmp_max_qval
                    self.q_table[str(piece_type)][state][1] = max_q_val

                self.q_table[str(piece_type)][state][0] = qval * (1-self.alpha) + self.alpha * self.gamma * max_q_val                

            tmp_max_qval = self.q_table[str(piece_type)][state][0]

    # a training function
    def learn(self, result):
        """ when games ended, this method will be called to update the qvalues
        """        
        if self.piece_type == result: # I win!  
            reward = WIN_REWARD
            opponent_reward = LOSS_REWARD
        elif 3-self.piece_type == result: # I lose
            reward = LOSS_REWARD
            opponent_reward = WIN_REWARD
        else:
            reward = TIE_REWARD
            opponent_reward = TIE_REWARD

        self.update_Q(self.piece_type, self.history_states, reward)

        self.update_Q(3-self.piece_type, self.opponent_history_states, opponent_reward)

        self.output_Q()

    def read_Q(self, infile='Qvalues.txt'): 
        with open(infile, 'r') as f:
            Q_vals = json.load(f)
        return Q_vals
 
    def output_Q(self, outfile='Qvalues.txt'):
        Q_vals = json.dumps(self.q_table, indent=4)
        with open(outfile, "w") as f:
            f.write(Q_vals)

####################################### utility functions copied ######################################
# Credits to Foundations of Artificial Intelligence teaching staff in 2022
#########################################################################
#                       Game Host Object                                #
#########################################################################
class GO:
    def __init__(self, n):
        """
        Go game.

        :param n: size of the board n*n
        """
        self.size = n
        #self.previous_board = None # Store the previous board
        #self.X_move = True # X chess plays first
        self.died_pieces = [] # Intialize died pieces to be empty
        # self.n_move = 0 # Trace the number of moves
        # self.max_move = n * n - 1 # The max movement of a Go game
        # self.komi = n/2 # Komi rule
        self.verbose = False # Verbose only when there is a manual player

    def set_board(self, piece_type, previous_board, board):
        '''
        Initialize board status.
        :param previous_board: previous board state.
        :param board: current board state.
        :return: None.
        '''

        # 'X' pieces marked as 1
        # 'O' pieces marked as 2

        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.died_pieces.append((i, j))

        # self.piece_type = piece_type
        self.previous_board = previous_board
        self.board = board

    def compare_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def copy_board(self):
        '''
        Copy the current board for potential testing.

        :param: None.
        :return: the copied board instance.
        '''
        return deepcopy(self)

    def detect_neighbor(self, i, j):
        '''
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i-1, j))
        if i < len(board) - 1: neighbors.append((i+1, j))
        if j > 0: neighbors.append((i, j-1))
        if j < len(board) - 1: neighbors.append((i, j+1))
        return neighbors

    def detect_neighbor_ally(self, i, j):
        '''
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = self.detect_neighbor(i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self, i, j):
        '''
        Using DFS to search for all allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the all allies row and column (row, column) of position (i, j).
        '''
        stack = [(i, j)]  # stack for DFS serach
        ally_members = []  # record allies positions during the search
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def find_liberty(self, i, j):
        '''
        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        '''
        board = self.board
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False

    def find_died_pieces(self, piece_type):
        '''
        Find the died stones that has no liberty in the board for a given piece type.

        :param piece_type: 1('X') or 2('O').
        :return: a list containing the dead pieces row and column(row, column).
        '''
        board = self.board
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.find_liberty(i, j):
                        died_pieces.append((i,j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        '''
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        '''

        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces: return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces

    def remove_certain_pieces(self, positions):
        '''
        Remove the stones of certain locations.

        :param positions: a list containing the pieces to be removed row and column(row, column)
        :return: None.
        '''
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_board(board)


    def valid_place_check(self, i, j, piece_type, test_check=False):
        '''
        Check whether a placement is valid.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1(white piece) or 2(black piece).
        :param test_check: boolean if it's a test check.
        :return: boolean indicating whether the placement is valid.
        '''   
        board = self.board
        verbose = self.verbose
        if test_check:
            verbose = False

        # Check if the place is in the board range
        if not (i >= 0 and i < len(board)):
            if verbose:
                print(('Invalid placement. row should be in the range 1 to {}.').format(len(board) - 1))
            return False
        if not (j >= 0 and j < len(board)):
            if verbose:
                print(('Invalid placement. column should be in the range 1 to {}.').format(len(board) - 1))
            return False
        
        # Check if the place already has a piece
        if board[i][j] != 0:
            if verbose:
                print('Invalid placement. There is already a chess in this position.')
            return False
        
        # Copy the board for testing
        test_go = self.copy_board()
        test_board = test_go.board

        # Check if the place has liberty
        test_board[i][j] = piece_type
        test_go.update_board(test_board)
        if test_go.find_liberty(i, j):
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(i, j):
            if verbose:
                print('Invalid placement. No liberty found in this position.')
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.compare_board(self.previous_board, test_go.board):
                if verbose:
                    print('Invalid placement. A repeat move not permitted by the KO rule.')
                return False
        return True
        
    def update_board(self, new_board):
        '''
        Update the board with new_board

        :param new_board: new board.
        :return: None.
        '''   
        self.board = new_board

        

#########################################################################
#                       Read and Write                                  #
#########################################################################
def readInput(n, path="input.txt"):

    with open(path, 'r') as f:
        lines = f.readlines()

        piece_type = int(lines[0])

        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n+1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n+1: 2*n+1]]

        return piece_type, previous_board, board


def writeOutput(result, path="output.txt"):
    res = ""
    if result == "PASS":
        res = "PASS"
    else:
        res += str(result[0]) + ',' + str(result[1])

    with open(path, 'w') as f:
        f.write(res)

def writePass(path="output.txt"):
    with open(path, 'w') as f:
        f.write("PASS")


if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = Qlearner(save_Q=False)
    action = player.get_input(go, piece_type)
    writeOutput(action)
