import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

class FoeLearner():
    
    def __init__(self,env,seed=None):
        
        self.max_iterations = 1000000
        self.gamma = 0.90
        self.min_epsilon = 0.001
        self.initial_epsilon = 1.0
        self.epsilon_schedule = np.linspace(start=self.initial_epsilon,stop=self.min_epsilon,num=1000000)
        self.current_epsilon = self.epsilon_schedule[0]
        self.initial_alpha = 0.9
        self.min_alpha = 0.01
        #self.alpha_schedule = np.linspace(start=self.initial_alpha,stop=self.min_alpha,num=1000000)
        self.alpha_schedule = np.logspace(start=0,stop=-2,num=self.max_iterations,base=10, endpoint=True)
        self.current_alpha = self.alpha_schedule[0]
        self.env = env
        self.nA = self.env.nA
        self.q_table = np.zeros((8,8,2,5,5))
        self.error_log = []
        self.iter_list = []
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
       
    
  
    def poss_to_int(self):
        poss = self.env.possesion
        if poss == 'Player A':
            poss_int = 0
        else:
            poss_int = 1
        return poss_int
    
    
    def solve_maximin(self,Q):
        game_matrix = np.array(Q,dtype=np.float).T
        game_matrix = np.concatenate((np.ones((Q.shape[1],1)),game_matrix),axis=1)
        constraint_bottom = np.eye(game_matrix.shape[0])*-1
        constraint_bottom = np.concatenate((np.zeros((Q.shape[1],1)),constraint_bottom),axis=1)
        game_matrix = np.concatenate((game_matrix,constraint_bottom),axis=0)
        a_1 = np.hstack((0,np.ones(Q.shape[1])))
        a_2 = np.hstack((0,-np.ones(Q.shape[1])))
        A = matrix(np.vstack((game_matrix,a_1,a_2)))
        b = matrix(np.hstack((np.zeros(A.size[0] - 2), [1, -1])))
        c = matrix(np.hstack(([-1], np.zeros(Q.shape[1]))))
        sol = solvers.lp(c,A,b, solver='glpk')
        return sol['primal objective']
        
    def e_greedy_action(self,old_p1_pos,old_p2_pos,old_poss_int):
        if self.current_epsilon >= np.random.rand():
            action = np.argmax(self.q_table[old_p1_pos,old_p2_pos,old_poss_int])//5
        else:
            action = np.random.randint(5)
        return action
    
    def train_loop(self):
        self.env.reset_env()
        done = False
        for i in range(self.max_iterations):
            
           
            old_poss_int = self.poss_to_int()
            old_p1_pos = self.env.player_a_pos - 1
            old_p2_pos = self.env.player_b_pos -1
            #action_1 = np.random.randint(5)
            action_1 = self.e_greedy_action(old_p1_pos,old_p2_pos,old_poss_int)
            action_2 = np.random.randint(5)
            old_q_entry = self.q_table[old_p1_pos,old_p2_pos,old_poss_int]
            old_q_entry_1 = self.q_table[old_p1_pos,old_p2_pos,old_poss_int,action_1,action_2]
            _, reward, done = self.env.step(action_1,action_2)
            new_poss_int = self.poss_to_int()
            new_p1_pos = self.env.player_a_pos - 1
            new_p2_pos = self.env.player_b_pos - 1
            reward_1 = reward[0]
            reward_2 = reward[1]
            optim = self.solve_maximin(old_q_entry)
            self.q_table[old_p1_pos,old_p2_pos,old_poss_int,action_1,action_2] = (1-self.current_alpha)*self.q_table[old_p1_pos,old_p2_pos,old_poss_int,action_1,action_2]+self.current_alpha*((1-self.gamma)*reward_1+self.gamma*optim)
            if (old_p1_pos,old_p2_pos,old_poss_int,action_1,action_2) == (2,1,1,3,0):
                new_q_entry = self.q_table[old_p1_pos,old_p2_pos,old_poss_int,action_1,action_2]
                error = np.abs(new_q_entry-old_q_entry_1)
                self.iter_list.append(i)
                self.error_log.append(error)
            self.current_alpha = self.alpha_schedule[i]
            self.current_epsilon = self.epsilon_schedule[i]
            if done:
                self.env.reset_env()
                done = False
                
            if i%10000 == 0:
                print('Iteration Number :',i)

    def error_plot(self):
        errs = np.array(self.error_log)
        it_num = np.array(self.iter_list)
        plt.plot(it_num,errs,linewidth=1)
        plt.title('Foe Q Learner')
        plt.ylabel('Iteration Number')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.xlabel('Q Value Error')
        plt.ylim((0,.5))
        plt.show()