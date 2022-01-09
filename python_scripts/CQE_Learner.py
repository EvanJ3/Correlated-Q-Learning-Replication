import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

class CQLearner():
    
    def __init__(self,env):
        
        self.max_iterations = 750000
        self.gamma = 0.90
        self.min_epsilon = 0.001
        self.initial_epsilon = 0.5
        self.epsilon_schedule = np.linspace(start=self.initial_epsilon,stop=self.min_epsilon,num=self.max_iterations)
        self.current_epsilon = self.epsilon_schedule[0]
        self.initial_alpha = 0.9
        self.min_alpha = 0.001
        self.alpha_schedule = np.linspace(start=self.initial_alpha,stop=self.min_alpha,num=self.max_iterations)
        #np.random.seed(seed)
        #self.alpha_schedule = np.logspace(start=0,stop=-1,num=self.max_iterations,base=10, endpoint=True)
        self.current_alpha = self.alpha_schedule[0]
        self.env = env
        self.nA = self.env.nA
        self.q_table_1 = np.zeros((8,8,2,5,5))
        self.q_table_2 = np.zeros((8,8,2,5,5))
        self.error_log = []
        self.iter_list = []
    
  
    def poss_to_int(self):
        poss = self.env.possesion
        if poss == 'Player A':
            poss_int = 0
        else:
            poss_int = 1
        return poss_int
    
    
    def corr_eq_solve(self,q1,q2):

        A = np.zeros((40,25))
        row_counter = 0
        for i in range(self.nA):
            for j in range(self.nA):
                if i != j:
                    A[row_counter, i * self.nA:(i + 1) * self.nA] = q1[i] - q1[j]
                    A[row_counter + self.nA * (self.nA - 1), i:(self.nA**2):self.nA] = q2[:, i] - q2[:, j]
                    row_counter += 1
                    
        
        '''s1 = block_diag(q1 - q1[0, :], q1 - q1[1, :], q1 - q1[2, :], q1 - q1[3, :], q1 - q1[4, :])
        row_index = range(0,25)
        for in row_index:
            if i % 5 == 0:
                row_index.remove(i)
        p1_con = s1[row_index, :]
        s2 = block_diag(q2 - q2[0, :], q2 - q2[1, :], q2 - q2[2, :], q2 - q2[3, :], q2 - q2[4, :])
        col_index = np.arange(0,25).reshape((5,5)).T.flatten()
        
        p2_con = s2[row_index, :][:, col_index]'''
        
        #A = np.concatenate((p1_con,p2_con),axis=0)
        
        A = np.hstack((np.ones((A.shape[0], 1)), A))
        eye_mat = np.hstack((np.zeros((25, 1)), -np.eye(25)))
        A = np.vstack((A, eye_mat))
        a_1 = np.hstack((0,np.ones(25)))
        a_2 = np.hstack((0,-np.ones(25)))
        A = matrix(np.vstack((A, a_1, a_2)))
        b = matrix(np.hstack((np.zeros(A.size[0] - 2), [1, -1])))
        c = matrix(np.hstack(([-1.], -(q1+q2).flatten())))
        sol = solvers.lp(c,A,b, solver='glpk')
        if sol['x'] is None:
            return 0, 0
        probs = sol['x'][1:]
        q1_vec = q1.flatten()
        q2_vec = q2.T.flatten()
        qv1 = q1_vec @ probs
        qv2 = q2_vec @ probs
        return qv1[0], qv2[0]

    def e_greedy_action(self,old_p1_pos,old_p2_pos,old_poss_int):
        if self.current_epsilon <= np.random.rand():
            action = np.argmax(self.q_table_1[old_p1_pos,old_p2_pos,old_poss_int])//5
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
            old_q_entry_1 = self.q_table_1[old_p1_pos,old_p2_pos,old_poss_int,action_1,action_2]
            _, reward, done = self.env.step(action_1,action_2)
            new_poss_int = self.poss_to_int()
            new_p1_pos = self.env.player_a_pos - 1
            new_p2_pos = self.env.player_b_pos - 1
            reward_1 = reward[0]
            reward_2 = reward[1]
            Q1 = self.q_table_1[old_p1_pos,old_p2_pos,old_poss_int,:,:]
            Q2 = self.q_table_2[old_p1_pos,old_p2_pos,old_poss_int,:,:]
            optim_1,optim_2 = self.corr_eq_solve(Q1,Q2)
            self.q_table_1[old_p1_pos,old_p2_pos,old_poss_int,action_1,action_2] = (1-self.current_alpha)*self.q_table_1[old_p1_pos,old_p2_pos,old_poss_int,action_1,action_2]+self.current_alpha*((1-self.gamma)*reward_1+self.gamma*optim_1)
            self.q_table_2[old_p1_pos,old_p2_pos,old_poss_int,action_1,action_2] = (1-self.current_alpha)*self.q_table_2[old_p1_pos,old_p2_pos,old_poss_int,action_1,action_2]+self.current_alpha*((1-self.gamma)*reward_2+self.gamma*optim_2)
            if (old_p1_pos,old_p2_pos,old_poss_int,action_1,action_2) == (2,1,1,3,0):
                new_q_entry = self.q_table_1[old_p1_pos,old_p2_pos,old_poss_int,action_1,action_2]
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
        plt.plot(np.array(self.error_log))
        plt.show()