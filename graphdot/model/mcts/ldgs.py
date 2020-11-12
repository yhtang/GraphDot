#!/usr/bin/env python
# -*- coding: utf-8 -*-


class LikelihoodDrivenGraphSearch:
    '''Monte Carlo tree search for creating graphs with desired propreties.

    Parameters
    ----------
    target: float
        Desired value for the target property.
    predictor: callable
        A predictor used to calculate the target property of a given graph.
    '''

    def __init__(self, rewriter, evaluator, score='ucb'):
        self.rewriter = rewriter
        self.evaluator = evaluator
        # self.tree = tree
        # self.iterations = iterations
        # self.depth = depth
        # self.width = width
        # self.target = target
        # self.predictor = predictor
        # self.margin = margin
        # self.sigma_margin = sigma_margin
        # self.exploration_constant = exploration_constant
        # self.scoring = scoring

    # def __str__(self):
    #     return str(self.tree)

    @staticmethod
    def argmax(iterable, less):
        best = None
        for i in iterable:
            if best is None or less(best, i):
                best = i
        return best

    def select_best_child(self, node):
        '''Recursively selects the optimal child among the children of a node.
        '''
        best = self.argmax(
            node.children,
            lambda i, j: self.score(i) < self.score(j)
        )
        return best if best.children is None else self.select_best_child(best)

    def expand(self, node):
        '''Create one or more children if the node is not terminal.'''
        assert node.children is None, f'Cannot expand non-leaf node {node}.'
        node.children = self.rewriter(node)

    def simulate(self, node):
        '''Select one or more children of a node and run simulation.'''
        mean, cov = self.evaluator(node.children)
        node.payloads

    def check_found(self, child, pred, sigma):
        ''' Checks to see if a node that fits the given constraints has been found.

        Parameters
        ----------
        child: TreeNode
            The input node that is being checked.
        pred: float
            Prediction of the node's associated value.
        sigma: float
            The node's associated uncertainty value.

        Returns
        -------
        found: boolean
            Boolean representing whether or not the node has an associated value
            within the acceptable margin of error.
        '''
        if sigma < self.sigma_margin and abs(pred - self.target) < self.margin:
            child.selected = True
            return True
        child.selected = False
        return False
    
    def selection(self): 
        ''' Select leaf node with maximum score.. 

        Returns
        -------
        max_node: TreeNode
            The node with maximum score.
        depth: integer
            The depth of the max_node found from the root node
        '''
        max_node = self.tree
        max_score = self.tree.score
        depth = 0
        while len(max_node.children) != 0: 
            prev_max = max_node
            depth += 1
            for child in max_node.children: 
                if child.score > max_score:
                    max_score = child.score
                    max_node = child
            if max_node == prev_max:
                children = np.array(max_node.children)
                max_node = children[np.random.randint(0, len(children), 1)[0]]
        return max_node, depth
    
    def expansion(self, node):
        ''' Create all possible outcomes from leaf node. 
        Parameters
        ----------
        node: TreeNode
            The node with the maximum score found by selection()

        Returns
        -------
        created: boolean
            Whether or not new children were added to the given node
        '''
        graph = node.graph
        for i in range(self.width): 
            try:
                r = Rewrite()
                new_graph = r(graph)
                child = TreeNode(children=[], parent=node, graph=new_graph, visits=0, allchild=[], depth=node.depth+1)
                node.children.append(child)
            except:
                pass
        return node.children
                
    def simulation(self, node):
        ''' Simulate game from child node's state until it reaches the resulting state of the game. 

        Parameters
        ----------
        node: TreeNode
            The child node found by expansion()

        Returns
        -------
        result: boolean, 
            Whether or not a new node was found.
        child: TreeNode
            The newly found node.
        graphs: list
            The list of graphs of all the node's children
        '''
        graph = node.graph
        graphs = [tograph(child.graph) for child in node.children]
        mean, cov = self.predictor.predict(graphs, return_cov=True)
        for i in range(len(node.children)):
            child = node.children[i]
            if self.scoring == "mlt":
                child.mean = mean[i]
                child.std = np.sqrt(cov[i][i])
                exploitation = norm(child.mean, child.std).pdf(self.target)
                exploration = np.sqrt(self.tree.visits)
                child.score = exploitation + self.exploration_constant * exploration
            else:
                # TODO: UCT implementation
                child.score = 0
            child.visits += 1
            if self.check_found(child, child.mean, child.std):
                return True, child, graphs
        return False, None, graphs
    
    def backpropagation(self, node, allchild_graphs):
        ''' Backpropagate against the nodes, updating values.

        Parameters
        ----------
        node: TreeNode
            The chosen node of maximum value
        allchild_graphs: list
            The list of graphs of all the given node's children
        '''
        while node != None:
            node.allchild.extend(allchild_graphs)
            node.visits += 1
            npmean, npvar = self.predictor.predict(node.allchild, return_cov=True)
            if self.scoring == "mlt":
                npvar.flat[::len(npvar) + 1] += 1e-10
                npvar_inv = np.linalg.inv(npvar)
                child_mean = (np.sum(npvar_inv@npmean))/np.sum(npvar_inv)
                child_sigma = np.sqrt(1 / np.sum(npvar_inv))
                exploitation = norm(child_mean, child_sigma).pdf(self.target)
                exploration = np.sqrt(np.log(self.tree.visits / node.visits))
                node.post_mean = child_mean
                node.post_var = child_sigma
                node.score = exploitation + self.exploration_constant * exploration
            else:
                # TODO: implement UCT
            node = node.parent

    def __call__(self):
        ''' Run MCTS and select best action. 

        Parameters
        ----------
        seed: graph
            The starting point of the Monte Carlo tree search.
        iterations: integer
            The number of MCTS iterations to run in each turn.
        depth: integer
            The maximum depth allowed in node selection.
        branching_factor: integer
            The maximum number of children for a node.
        target_confidence: float
            Only graphs that lies with the given confidence interval 
            The acceptable margin of error around the desired value for the target
            property.
        sigma_margin: float
            The largest degree of uncertainty that can be accepted.
        exploration_constant: float
            A tunable hyperparameter that controls the balance between exploration
            and exploitation.
        score: 'uct' or 'l4t'
            The scoring function to be used for node selection. 'uct' = upper
            confidence bounds applied to trees. 'l4t' = likelihood applied to
            trees.


        Returns
        -------
        found: boolean
            Whether or not a node fitting the given constraints has been found.
        max_node: TreeNode
            The node with the maximum score in the tree.
        '''
        for i in range(self.iterations):
            max_node, depth = self.selection()
            if depth >= self.depth:
                break
            created = self.expansion(max_node)
            if not created: # No more possible actions can be made
                break
            found, child, allchild_graphs = self.simulation(max_node)
            if found: # Node found
                return found, child
            self.backpropagation(max_node, allchild_graphs)
            max_node.selected = True
        max_node = self.tree
        max_score = self.tree.score
        while len(max_node.children) != 0: 
            prev_max = max_node
            for child in max_node.children: 
                if child.score > max_score:
                    max_score = child.score
                    max_node = child
            if max_node == prev_max:
                break
        return False, max_node

