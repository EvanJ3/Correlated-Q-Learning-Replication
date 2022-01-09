import numpy as np
import matplotlib.pyplot as plt
        
class QLearner():
    
    def __init__(self,env,seed=None,lin=True):
        
        self.max_iterations = 1000000
        self.gamma = 0.90
        self.min_epsilon = 0.001
        self.initial_epsilon = 1.0
        self.epsilon_schedule = np.linspace(start=self.initial_epsilon,stop=self.min_epsilon,num=1000000)
        #self.epsilon_schedule = np.repeat([1.0,0.001],[750000,250000])
        #self.epsilon_schedule = np.logspace(start=0,stop=-3,num=self.max_iterations,base=10, endpoint=True)
        self.current_epsilon = self.epsilon_schedule[0]
        self.initial_alpha = 1.0
        self.min_alpha = 0.001
        if lin:  
            self.alpha_schedule = np.linspace(start=self.initial_alpha,stop=self.min_alpha,num=self.max_iterations)
        else:
            self.alpha_schedule = np.logspace(start=0,stop=-2,num=self.max_iterations,base=10, endpoint=True)
        self.current_alpha = self.alpha_schedule[0]
        self.env = env
        self.nA = self.env.nA
        self.q_table = np.zeros((8,8,2,5))
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
        
    def e_greedy_action(self,old_p1_pos,old_p2_pos,old_poss_int):
        if self.current_epsilon >= np.random.rand():
            action = np.argmax(self.q_table[old_p1_pos,old_p2_pos,old_poss_int])
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
            action_1 = self.e_greedy_action(old_p1_pos,old_p2_pos,old_poss_int)
            action_2 = np.random.randint(5)
            old_q_entry = self.q_table[old_p1_pos,old_p2_pos,old_poss_int,action_1]
            _, reward, done = self.env.step(action_1,action_2)
            new_poss_int = self.poss_to_int()
            new_p1_pos = self.env.player_a_pos - 1
            new_p2_pos = self.env.player_b_pos - 1
            reward_1 = reward[0]
            reward_2 = reward[1]
            self.q_table[old_p1_pos,old_p2_pos,old_poss_int,action_1] = (1-self.current_alpha)*self.q_table[old_p1_pos,old_p2_pos,old_poss_int,action_1]+self.current_alpha*((1-self.gamma)*reward_1+self.gamma*np.max(self.q_table[new_p1_pos,new_p2_pos,new_poss_int,:]))
            if (old_p1_pos,old_p2_pos,old_poss_int,action_1,action_2) == (2,1,1,3,0):
                new_q_entry = self.q_table[old_p1_pos,old_p2_pos,old_poss_int,action_1]
                error = np.abs(new_q_entry-old_q_entry)
                self.error_log.append(error)
                self.iter_list.append(i)
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
        
