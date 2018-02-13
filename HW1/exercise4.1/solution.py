import networkx as nx
import argparse

# Determine if nodes have strong or weak tie
def haveStrongTie(node1, node2):

    # Read in edges
    G = nx.read_edgelist("./CA-GrQc.txt")

    # Get neighbor sets
    node1_neighbors = set(list(G.neighbors(node1)))
    node2_neighbors = set(list(G.neighbors(node2)))

    # Matching coauthor
    if node1_neighbors & node2_neighbors:
        return True

    # # Share an edge
    # if node1 in node2_neighbors or node2 in node1_neighbors:
    #     return True

    # Weak tie / No edge
    return False

# Parse input args
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('node1', type=str)
    parser.add_argument('node2', type=str)

    args = parser.parse_args()

    returnVal = haveStrongTie(args.node1, args.node2)
    print(returnVal)
