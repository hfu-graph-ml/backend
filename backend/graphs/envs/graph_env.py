import random
import gym
import copy
import numpy as np
import networkx as nx
from utils import draw_graph

from config import config
from evaluation.evaluation import valid_table_graph
from data.edge_score_matrix import get_edge_scores


class GraphEnv(gym.Env):
    '''
    This class describes the environment that the agent interacts with and gets its feedback from (in form of rewards).
    It holds a representation of the seating plan graph and updates it according to the agent's actions.

    *Note: Please use the `init` method to initialize the environment correctly.*
    '''

    def __init__(self):
        # Legacy initialization function without args.
        # *Do not use this function! Please use the `init` method instead.*
        pass

    def init(self, base_graph: nx.Graph, preconnect_nodes_probability=0):
        '''
        Initializes the environment.

        Args:
            base_graph: The seating plan graph the environment starts with and is reset to. It should contain all nodes and may also have edges.
            preconnect_nodes_probability (`float`, optional): The probability with wich a random number of edges get added the seating plan graph after resetting the environment. Defaults to 0.
        '''
        self.config = config
        self.base_graph = base_graph
        self.graph = copy.deepcopy(self.base_graph)
        self.preconnect_nodes_probability = preconnect_nodes_probability

        self.counter = 0

        self.num_nodes = self.base_graph.number_of_nodes()
        self.max_edges = (((self.num_nodes * 3) - 4) / 2)

        self.action_space = gym.spaces.MultiDiscrete([self.num_nodes, self.num_nodes])
        self.observation_space = gym.spaces.Dict({
                'adj': gym.spaces.Box(low=0, high=self.num_nodes, shape=(1, self.num_nodes, self.num_nodes), dtype=np.uint8),
                'node': gym.spaces.Box(low=0, high=self.num_nodes, shape=(1, self.num_nodes, self.config['data']['num_node_features']), dtype=np.uint8)
        })

        self.level = 0  # for curriculum learning, level starts with 0, and increase afterwards

        # draw graph
        if self.config['debugging']['draw_graph']:
            draw_graph(self.graph, layout=None)

    def step(self, action):
        '''
        Updates the environment according to the given `action` if it's valid. Sets rewards for the agent. Once the graph has enough edges, it sets terminal signal that resets the environment.

        Args:
            action: The action to manipulate the seating plan graph. An array holding two integers for the IDs of the nodes to connect.

        Returns:
            ob: The observation of the seating plan graph as `dict` with adjacency matrix (adj) and node matrix with features (node).
            reward: The reward calculated in the current step.
            new: The terminal signal to reset the environment.
            info: A dict with useful information for evaluation.
        '''
        # init
        info = {}  # info we care about

        graph_is_valid = None
        mood_scores = []

        # take action or not
        edge_added = self._add_edge(action)

        # print actions
        if self.config['debugging']['print_actions']:
            print(action)

        # get observation
        ob = self.get_observation()

        # wheter to stop after this step
        stop = self.graph.number_of_edges() >= (
                ((self.config['data']['num_nodes'] * 3) - 4) / 2) or self.counter >= self.config['training']['max_steps']

        # calculate intermediate rewards
        if edge_added:
            reward_step = self.config['rewards']['step_edge_correct']
        else:
            reward_step = self.config['rewards']['step_edge_incorrect']

        # calculate and use terminal reward
        if stop:
            new = True  # end of episode

            graph_is_valid = valid_table_graph(self.graph)

            if graph_is_valid:
                mood_scores = get_edge_scores(self.graph.edges())
                reward_terminal = (np.sum(mood_scores) - self.config['rewards']['score_min']) / (
                        self.config['rewards']['score_max'] - self.config['rewards']['score_min']) * self.config['rewards']['terminal_valid_score_multiplier']

                # draw finalized graph
                if self.config['debugging']['draw_correct_graphs']:
                    print(reward_terminal)
                    draw_graph(self.graph)

            else:
                reward_terminal = self.config['rewards']['terminal_invalid']

            reward = reward_step + reward_terminal
            # print terminal graph information
            info['final_stat'] = reward_terminal
            info['reward'] = reward
            info['stop'] = stop

        # use stepwise reward
        else:
            new = False
            reward = reward_step

        self.counter += 1
        if new:
            self.counter = 0

        info['graph'] = self.graph
        info['action_valid'] = int(edge_added)
        info['graph_valid'] = int(graph_is_valid) if graph_is_valid != None else -np.inf
        info['mood_score'] = np.sum(mood_scores) if len(mood_scores) > 0 else np.nan

        if self.config['debugging']['print_actions']:
            print(reward)

        # draw graph
        if self.config['debugging']['draw_graph']:
            draw_graph(self.graph, layout=None)

        return ob, reward, new, info

    def reset(self):
        '''
        Resets the environment's seating plan graph to given `self.base_graph` and with the probability set by `self.preconnect_nodes_probability` adds random number of edges (between 1 and `max_edges` - 1).
        '''
        self.graph = copy.deepcopy(self.base_graph)

        # preconnecting nodes simulates user input
        if random.random() < self.preconnect_nodes_probability:

            # create list of shuffled node ids
            graph_nodes = np.array(self.graph)
            random.shuffle(graph_nodes)

            # generate grid graph to get valid table structure
            grid_graph = nx.grid_graph((2, len(graph_nodes)//2))

            # convert positional ids from grid graph eg. (2, 1) to indices eg. 12
            # use these indices in shuffeled node array to randomly assign new id
            # and generate edge list for the random table graph
            graph_edges = [(graph_nodes[edge[0][0]+len(graph_nodes)//2*edge[0][1]], graph_nodes[edge[1][0]+len(graph_nodes)//2*edge[1][1]])
                                         for edge in grid_graph.edges]

            # only add some random amount of edges to achieve incomplete table graph
            self.graph.add_edges_from(random.sample(graph_edges, k=random.randint(1, len(graph_edges)-1)))

            if self.config['debugging']['draw_preconnected_graphs']:
                print("preconnected graph")
                draw_graph(self.graph)

        self.counter = 0
        ob = self.get_observation()
        return ob

    def render(self):
        return

    def _add_edge(self, action):
        """
        Adds only valid edges between the given nodes.

        Args:
            action: Array of two nodes to connect.
        
        Returns:
            True if valid edge was added successfully. False if invalid edge wasn't added successfully.
        """

        if self.graph.has_edge(int(action[0]), int(action[1])) or int(action[0]) == int(action[1]) or self.graph.degree(action[0]) >= 3 or self.graph.degree(action[1]) >= 3:
            return False
        else:
            graph_old = copy.deepcopy(self.graph)
            self.graph.add_edge(int(action[0]), int(action[1]))
            if not nx.is_bipartite(self.graph):
                self.graph = graph_old
                return False
            return True

    def get_observation(self):
        """
        Converts the seating plan graph with type `nx.Graph` to dict with adjacency matrix and node matrix with features.

        Returns:
            ob: Observation dict where ob['adj'] is E with dim 1 x n x n and ob['node'] is F with dim 1 x n x m.
        """

        ob = {}
        ob['adj'] = np.expand_dims(nx.adjacency_matrix(self.graph).todense(), axis=0)
        ob['node'] = np.expand_dims(np.array([[feature for feature in node[1].values()]
                                                                for node in self.graph.nodes(data=True)]), axis=0)

        return ob
