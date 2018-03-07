#!/usr/local/bin/python3
import networkx as nx
import pandas as pd
import numpy as np
import dateutil.parser

def simulate(G, seedSetA, seedSetB, conversionRate):
    """
    Using Graph G and initial seed set, simulate network response to buy product
    using conversionRate. Calculates total profit = total number of nodes buying
    the product.

    Seed Set is otherwise called Set A. Set B is any node not in Set A

    Everyone of label A has received the product, buys it and promotes
    it to all of their neighbors in the graph.

    Everyone of label B buys the product with probability p. If they buy the product,
    they also promote it to all of their friends; if they don’t, they won’t promote it,
    so the cascade ends at that node.
    - If someone has bought the product, they won’t buy it or promote it again.

    Your profit is the number of people of who bought the product when this process is over.

    Input:
        G: networkx graph
        seedSetA: set of A nodes initially given product
        seedSetB: set of B nodes initially given product
        conversionRate: rate that set B will buy product
    Output:
        profit: integer

    Returns:
        Total profit (total number of users who purchased product)
    """

    # Mark all seed set nodes A
    # Didn't really use this
    # G.set_node_attributes(G, 'isA', dict(zip(seedSet,[1]*len(seedSet))))

    # Activate seed nodes
    combinedSet = np.append(seedSetA, seedSetB)
    promotingNodes = set(combinedSet)
    nodesWithProduct = set(combinedSet)

    # Don't need this since all A/B seeds handled the same
    # # Add all A nodes
    # promotingNodes = set(seedSetA)
    # nodesWithProduct = set(seedSetA)

    # B nodes in seed set are not handled separately
    # Handle B nodes separately
    # for seedB in seedSetB:
    #     nodeProb = np.random.rand()
    #
    #     # Converted B to purchase product
    #     # It promotes it to it's neighbors
    #     if nodeProb > conversionRate:
    #         promotingNodes.add(seedB)
    #         nodesWithProduct.add(seedB)

    # Keep updating list of nodes that are promoting or who have purchase the product
    while promotingNodes:

        # Evaluate all neighbors
        node = promotingNodes.pop()
        try:
            neighbors = G.neighbors(node)
        except nx.exception.NetworkXError:
            print("Using node %d, %d" % (node, node))

        for n in neighbors:

            # All nodes who haven't promoted yet
            if n not in nodesWithProduct:

                # setA node, added property for A nodes
                if G.node[n]!={}:
                    promotingNodes.add(n)
                    nodesWithProduct.add(n)

                else:
                    nodeProb = np.random.rand()

                    # Converted B to purchase product
                    # It promotes it to it's neighbors
                    if nodeProb < conversionRate:
                        # print("\t\tNode %s added: %s" % (node, n))
                        promotingNodes.add(n)
                        nodesWithProduct.add(n)

    # Calculate profit
    profit = len(nodesWithProduct)

    return profit
