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


if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = Qlearner(save_Q=False)
    action = player.get_input(go, piece_type)
    writeOutput(action)
