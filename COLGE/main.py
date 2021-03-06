import argparse
import agent
import environment
import runner
import graph
import logging
import numpy as np
import networkx as nx
import sys

# # 2to3 compatibility
# try:
#     input = raw_input
# except NameError:
#     pass

# Set up logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

parser = argparse.ArgumentParser(description='RL running machine')
parser.add_argument('--environment_name', metavar='ENV_CLASS', type=str, default='MVC', help='Class to use for the environment. Must be in the \'environment\' module')
parser.add_argument('--agent', metavar='AGENT_CLASS', default='Agent', type=str, help='Class to use for the agent. Must be in the \'agent\' module.')
parser.add_argument('--graph_type',metavar='GRAPH', default='erdos_renyi',help ='Type of graph to optimize')
parser.add_argument('--graph_nbr', type=int, default='1000', help='number of differente graph to generate for the training sample')
parser.add_argument('--model', type=str, default='S2V_QN_1', help='model name')
parser.add_argument('--ngames', type=int, metavar='n', default='500', help='number of games to simulate')
parser.add_argument('--niter', type=int, metavar='n', default='1000', help='max number of iterations per game')
parser.add_argument('--epoch', type=int, metavar='nepoch',default=25, help="number of epochs")
parser.add_argument('--lr',type=float, default=1e-4,help="learning rate")
parser.add_argument('--bs',type=int,default=32,help="minibatch size for training")
parser.add_argument('--n_step',type=int, default=3,help="n step in RL")
parser.add_argument('--node', type=int, metavar='nnode',default=20, help="number of node in generated graphs")
parser.add_argument('--p',default=0.14,help="p, parameter in graph degree distribution", type=float)
parser.add_argument('--m',default=4, type=int, help="m, parameter in graph degree distribution")
parser.add_argument('--batch', type=int, metavar='nagent', default=None, help='batch run several agent at the same time')
parser.add_argument('--verbose', action='store_true', default=True, help='Display cumulative results at each step')
parser.add_argument('--path', type=str, default='', metavar='path', help='Folder location of where to save results')
parser.add_argument('--hide_opt', action='store_true', default=False, help='Whether to calculate the optimal solution during of instances during training')
parser.add_argument('--gamma', type=float, default=0.99999, help='Discount factor')
parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay factor for learning rate scheduler')
parser.add_argument('--load_model', action='store_true', default=False, help='Whether to load a previously computed model')
parser.add_argument('--nvalid', type=int, default='50', help='number of graphs to use for the validation set')
parser.add_argument('--valid', action='store_true', default=False, help='Whether to just test the performance of the network, ideally used with loading the model')

def main():
    args = parser.parse_args()
    logging.info('Loading graph %s' % args.graph_type)
    graph_dic = {}
    #seed = 125
    #graph_one = graph.Graph(graph_type=args.graph_type, cur_n=20, p=0.15,m=4, seed=seed)

    for graph_ in range(args.graph_nbr):
        seed = np.random.seed(120+graph_)
        graph_dic[graph_]=graph.Graph(graph_type=args.graph_type, cur_n=args.node, p=args.p,m=args.m,seed=seed)

    logging.info('Loading agent...')
    agent_class = agent.Agent(graph_dic, args.model, args.lr,args.bs,args.n_step, args.gamma, args.lr_decay, args.path, args.load_model)

    logging.info('Loading environment %s' % args.environment_name)
    env_class = environment.Environment(graph_dic,args.environment_name)

    if args.batch is not None:
        print("Running a batched simulation with {} agents in parallel...".format(args.batch))
        my_runner = runner.BatchRunner(env_class, agent_class, args.batch, args.verbose)
        final_reward = my_runner.loop(args.ngames,args.epoch, args.niter)
        print("Obtained a final average reward of {}".format(final_reward))
        agent_class.save_model()
    else:
        print("Running a single instance simulation...")
        my_runner = runner.Runner(env_class, agent_class, args.path, args.verbose, args.hide_opt, args.nvalid)
        if args.valid:
            print("Testing a validation set...")
            my_runner.valid_loop(args.nvalid, args.niter)
        else:
            final_reward = my_runner.loop(args.ngames,args.epoch, args.niter)
            print("Obtained a final reward of {}".format(final_reward))
            agent_class.save_model()



if __name__ == "__main__":
    main()
