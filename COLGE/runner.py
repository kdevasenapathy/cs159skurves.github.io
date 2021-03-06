"""
This is the machinnery that runs your agent in an environment.

"""
import matplotlib.pyplot as plt
import numpy as np
import agent
import pathlib
import time

class Runner:
    def __init__(self, environment, agent, folder_path, verbose=False, hide_opt=False, n_valid=50):
        self.environment = environment
        self.agent = agent
        self.verbose = verbose
        self.hide_opt = hide_opt
        self.relative_folder_path = folder_path + '/'
        current_path = pathlib.Path().absolute()
        abs_folder_path = current_path / (folder_path + '/')
        self.abs_folder_path = abs_folder_path
        print(abs_folder_path)
        abs_folder_path.mkdir(parents=True, exist_ok=True)

    def step(self, validation=False):
        observation = self.environment.observe().clone()
        action = self.agent.act(observation, validation=validation).copy()
        (reward, done) = self.environment.act(action)
        self.agent.reward(observation, action, reward,done, validation=validation)
        return (observation, action, reward, done)
        
    def valid_loop(self, games, max_iter, save_file=True):
        
        games_to_run = []
        
        list_cumul_reward=[]
        list_optimal_ratio = []
        list_aprox_ratio = []
        
        num_avail_graphs = self.environment.get_number_of_stored_graphs()
        for g in range(num_avail_graphs - 1, num_avail_graphs -1 - games, -1):
            games_to_run.append(g)
        
        np.random.shuffle(games_to_run)
        
        num_games_ran = 0
        
        for g in games_to_run:
            num_games_ran += 1
            print(" -> Validation Game: {} \tRunning graph: {} : ".format(num_games_ran, g))
     
            self.environment.reset(g)
            self.agent.reset(g)
            cumul_reward = 0.0
            
            start_time = time.time()
            
            for i in range(1, max_iter + 1):
                (obs, act, rew, done) = self.step(validation=True)
                cumul_reward += rew
                
                if self.verbose and done:
                    end_time = time.time()
                    if not self.hide_opt:
                        approx_sol =self.environment.get_approx()
                        optimal_sol = self.environment.get_optimal_sol()         
                        
                        approx_ratio = max(abs(approx_sol/cumul_reward), abs(cumul_reward/approx_sol))
                        opt_ratio = max(abs(optimal_sol/cumul_reward), abs(cumul_reward/optimal_sol))
                        
                        print(" ->    Terminal event: cumulative rewards = {}\t opt = {}\t opt_ratio = {}\tTook {} seconds".format(cumul_reward, optimal_sol, opt_ratio, end_time-start_time))
                       
                        list_optimal_ratio.append(opt_ratio)
                        list_aprox_ratio.append(approx_ratio)
                    else:
                        # print cumulative reward of one play, it is actually the solution found by the NN algorithm
                        print(" ->    Terminal event: cumulative rewards = {}\tTook {} seconds".format(cumul_reward, end_time-start_time))
                    
                    #we add in a list the solution found by the NN algorithm
                    list_cumul_reward.append(-cumul_reward)
                    
                if done:
                    break
            if save_file:
                np.savetxt(self.relative_folder_path + 'valid_set.out', list_optimal_ratio, delimiter=',')

    def loop(self, games,nbr_epoch, max_iter):

        cumul_reward = 0.0
        list_cumul_reward=[]
        list_optimal_ratio = []
        list_aprox_ratio =[]

        for epoch_ in range(nbr_epoch):
            print(" -> epoch : "+str(epoch_))
            
            games_to_run = []
            
            for g in range(1, games + 1):
                for epoch in range(5):
                    games_to_run.append(g)
            
            np.random.shuffle(games_to_run)
            #print(games_to_run)
            num_games_ran = 0
            
            for g in games_to_run:
                num_games_ran += 1
                print(" -> Game: {} \tRunning graph: {} : ".format(num_games_ran, g))
            
            
                self.environment.reset(g)
                self.agent.reset(g)
                cumul_reward = 0.0
                
                start_time = time.time()
                
                for i in range(1, max_iter + 1):
                    # if self.verbose:
                    #   print("Simulation step {}:".format(i))
                    (obs, act, rew, done) = self.step()
                    cumul_reward += rew
                    if self.verbose:
                        #print(" ->       observation: {}".format(obs))
                        #print(" ->            action: {}".format(act))
                        #print(" ->            reward: {}".format(rew))
                        #print(" -> cumulative reward: {}".format(cumul_reward))
                        if done:
                            end_time = time.time()

                            if not self.hide_opt:
                                #solution from baseline algorithm
                                approx_sol =self.environment.get_approx()

                                #optimal solution
                                optimal_sol = self.environment.get_optimal_sol()         
                                
                                #print optimal solution
                                # print cumulative reward of one play, it is actually the solution found by the NN algorithm
                                print(" ->    Terminal event: cumulative rewards = {}\t opt = {}\t opt_ratio = {}\tTook {} seconds".format(cumul_reward, optimal_sol, max(cumul_reward/optimal_sol, optimal_sol/cumul_reward), end_time-start_time) )
                                print('LEARNING RATE: {}\t EPSILON: {}'.format(self.agent.optimizer.param_groups[0]['lr'], self.agent.epsilon_))
                                #we add in a list the ratio between the NN solution and the optimal solution
                                list_optimal_ratio.append(cumul_reward/(optimal_sol))

                                #we add in a list the ratio between the NN solution and the baseline solution
                                list_aprox_ratio.append(cumul_reward/(approx_sol))
                            else:
                                # print cumulative reward of one play, it is actually the solution found by the NN algorithm
                                print(" ->    Terminal event: cumulative rewards = {}\tTook {} seconds".format(cumul_reward, end_time-start_time))
                            
                            #we add in a list the solution found by the NN algorithm
                            list_cumul_reward.append(-cumul_reward)

                            

                    if done:
                        break
                np.savetxt(self.relative_folder_path + 'test_'+str(epoch_)+'.out', list_optimal_ratio, delimiter=',')
                np.savetxt(self.relative_folder_path + 'test_approx_' + str(epoch_) + '.out', list_aprox_ratio, delimiter=',')
                np.savetxt(self.relative_folder_path + 'opt_set.out', list_optimal_ratio, delimiter=',')

            #if self.verbose:
                #print(" <=> Finished game number: {} <=>".format(g))
                #print("")
                
            self.agent.scheduler.step()
            
            
        np.savetxt(self.relative_folder_path + 'test.out', list_cumul_reward, delimiter=',')
        np.savetxt(self.relative_folder_path + 'opt_set.out', list_optimal_ratio, delimiter=',')
        #plt.plot(list_cumul_reward)
        #plt.show()
        return cumul_reward

def iter_or_loopcall(o, count):
    if callable(o):
        return [ o() for _ in range(count) ]
    else:
        # must be iterable
        return list(iter(o))

class BatchRunner:
    """
    Runs several instances of the same RL problem in parallel
    and aggregates the results.
    """

    def __init__(self, env_maker, agent_maker, count, verbose=False):
        self.environments = iter_or_loopcall(env_maker, count)
        self.agents = iter_or_loopcall(agent_maker, count)
        assert(len(self.agents) == len(self.environments))
        self.verbose = verbose
        self.ended = [ False for _ in self.environments ]

    def game(self, max_iter):
        rewards = []
        for (agent, env) in zip(self.agents, self.environments):
            env.reset()
            agent.reset()
            game_reward = 0
            for i in range(1, max_iter+1):
                observation = env.observe()
                action = agent.act(observation)
                (reward, stop) = env.act(action)
                agent.reward(observation, action, reward)
                game_reward += reward
                if stop :
                    break
            rewards.append(game_reward)
        return sum(rewards)/len(rewards)

    def loop(self, games,nb_epoch, max_iter):
        cum_avg_reward = 0.0
        for epoch in range(nb_epoch):
            for g in range(1, games+1):
                avg_reward = self.game(max_iter)
                cum_avg_reward += avg_reward
                if self.verbose:
                    print("Simulation game {}:".format(g))
                    print(" ->            average reward: {}".format(avg_reward))
                    print(" -> cumulative average reward: {}".format(cum_avg_reward))
        return cum_avg_reward
