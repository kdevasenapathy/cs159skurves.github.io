import numpy as np
import torch
import pulp
import networkx as nx

from algorithms.feedbackVertex.maximum_induced_forest import MaximumInducedForest



"""
This file contains the definition of the environment
in which the agents are run.
"""


class Environment:
    def __init__(self, graph,name):
        self.graphs = graph
        self.name= name

    def reset(self, g):
        self.games = g
        self.graph_init = self.graphs[self.games]
        self.nodes = self.graph_init.nodes()
        self.nbr_of_nodes = 0
        self.edge_add_old = 0
        self.last_reward = 0
        self.observation = torch.zeros(1,self.nodes,1,dtype=torch.float)

    def observe(self):
        """Returns the current observation that the agent can make
                 of the environment, if applicable.
        """
        return self.observation

    def act(self,node):

        self.observation[:,node,:]=1
        reward = self.get_reward(self.observation, node)
        return reward

    def get_reward(self, observation, node):

        if self.name == "MVC":

            new_nbr_nodes=np.sum(observation[0].numpy())

            if new_nbr_nodes - self.nbr_of_nodes > 0:
                reward = -1#np.round(-1.0/20.0,3)
            else:
                reward = 0

            self.nbr_of_nodes=new_nbr_nodes

            #Minimum vertex set:

            done = True

            edge_add = 0

            for edge in self.graph_init.edges():
                if observation[:,edge[0],:]==0 and observation[:,edge[1],:]==0:
                    done=False
                    # break
                else:
                    edge_add += 1

            #reward = ((edge_add - self.edge_add_old) / np.max(
            #   [1, self.graph_init.average_neighbor_degree([node])[node]]) - 10)/100

            self.edge_add_old = edge_add

            return (reward,done)

        elif self.name=="MAXCUT" :

            reward=0
            done=False

            adj= self.graph_init.edges()
            select_node=np.where(self.observation[0, :, 0].numpy() == 1)[0]
            for nodes in adj:
                if ((nodes[0] in select_node) & (nodes[1] not in select_node)) | ((nodes[0] not in select_node) & (nodes[1] in select_node))  :
                    reward += 1#/20.0
            change_reward = reward-self.last_reward
            if change_reward<=0:
                done=True

            self.last_reward = reward

            return (change_reward,done)

        #Minimum Feedback Vertex Set: https://en.wikipedia.org/wiki/Feedback_vertex_set
        #Implemntation: https://github.com/chinhodado/CSI4105-Project
        elif self.name=="MFVS":

          ### Penalize each new vertex which is cut
            new_nbr_nodes=np.sum(observation[0].numpy())

            if new_nbr_nodes - self.nbr_of_nodes > 0:
                reward = -1
            else:
                reward = 0

            self.nbr_of_nodes=new_nbr_nodes

            list_of_nodes = []
            for node in range(self.graph_init.nodes()):
                if observation[:, node, :] == 0:
                    list_of_nodes.append(node)
            total_g = self.graph_init.get_graph()
            done = False
            #num_cycles = nx.find_cycle(total_g.subgraph(list_of_nodes))
            #print("Number of edges in cycle: " + str(num_cycles))
            try:
                num_cycles = nx.find_cycle(total_g.subgraph(list_of_nodes))
                #print("Number of edges in cycle: " + str(num_cycles))
            except:
                done = True

            return (reward, done)

        # INDEPENDENT SET
        #**************************************************************
        elif self.name=="IS":
            reward=1
            done = False

            # basically the anticlique
            clique = np.where(self.observation[0,:,0].numpy() == 1)[0]
            not_clique   = np.where(self.observation[0,:,0].numpy() == 0)[0]

            # want to make sure that there is at least one node not connected
            # to any of the other nodes we have selected

            for next_node in not_clique:
                for node in clique:
                    if self.gra:

            return (reward, done)



    def get_approx(self):

        if self.name=="MVC":
            cover_edge=[]
            edges= list(self.graph_init.edges())
            while len(edges)>0:
                edge = edges[np.random.choice(len(edges))]
                cover_edge.append(edge[0])
                cover_edge.append(edge[1])
                to_remove=[]
                for edge_ in edges:
                    if edge_[0]==edge[0] or edge_[0]==edge[1]:
                        to_remove.append(edge_)
                    else:
                        if edge_[1]==edge[1] or edge_[1]==edge[0]:
                            to_remove.append(edge_)
                for i in to_remove:
                    edges.remove(i)
            return len(cover_edge)

        elif self.name=="MAXCUT":
            return 1
        elif self.name=="MFVS":
            return self.graph_init.nodes()

        #**************************************************************
        elif self.name=="IS":
            return 1
        else:
            return 'you pass a wrong environment name'

    def get_optimal_sol(self):

        if self.name =="MVC":

            x = list(range(self.graph_init.g.number_of_nodes()))
            xv = pulp.LpVariable.dicts('is_opti', x,
                                       lowBound=0,
                                       upBound=1,
                                       cat=pulp.LpInteger)

            mdl = pulp.LpProblem("MVC", pulp.LpMinimize)

            mdl += sum(xv[k] for k in xv)

            for edge in self.graph_init.edges():
                mdl += xv[edge[0]] + xv[edge[1]] >= 1, "constraint :" + str(edge)
            mdl.solve()

            #print("Status:", pulp.LpStatus[mdl.status])
            optimal=0
            for x in xv:
                optimal += xv[x].value()
                #print(xv[x].value())
            return optimal

        elif self.name=="MAXCUT":

            x = list(range(self.graph_init.g.number_of_nodes()))
            e = list(self.graph_init.edges())
            xv = pulp.LpVariable.dicts('is_opti', x,
                                       lowBound=0,
                                       upBound=1,
                                       cat=pulp.LpInteger)
            ev = pulp.LpVariable.dicts('ev', e,
                                       lowBound=0,
                                       upBound=1,
                                       cat=pulp.LpInteger)

            mdl = pulp.LpProblem("MVC", pulp.LpMaximize)

            mdl += sum(ev[k] for k in ev)

            for i in e:
                mdl+= ev[i] <= xv[i[0]]+xv[i[1]]

            for i in e:
                mdl+= ev[i]<= 2 -(xv[i[0]]+xv[i[1]])

            #pulp.LpSolverDefault.msg = 1
            mdl.solve()

            # print("Status:", pulp.LpStatus[mdl.status])

            return mdl.objective.value()

        elif self.name == "MFVS":
            mif = MaximumInducedForest()
            g = self.graph_init.get_graph()
            g_copy = g.copy()
            return max(len(mif.get_fbvs(g_copy)), 1)

        # INDEPENDENT SET
        #**************************************************************
         elif self.name=="IS":
            return 0


