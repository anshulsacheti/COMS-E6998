import networkx as nx

# Determine if nodes have strong or weak tie
def haveStrongTie(G, node1, node2):

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


# Read in edges
G = nx.read_edgelist("./CA-GrQc.txt")

# Calculate number of strong/weak ties
strongTies = 0
weakTies = 0

nodes = list(G.nodes)

# Iterate over every edge
# Determine if edge is a strong tie or weak tie
edgeSet = nx.edges(G).keys()
for edge in edgeSet:
    n1 = edge[0]
    n2 = edge[1]

    if haveStrongTie(G, n1, n2):
        strongTies+=1
    else:
        weakTies+=1

edges = G.number_of_edges()
print("Number of edges: %d" % edges)
print("Number of strong ties: %d" % strongTies)
print("Number of weak ties: %d" % weakTies)
