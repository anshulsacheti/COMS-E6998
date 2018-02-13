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

origCC = nx.number_connected_components(G)
maxOrigCC = len(max(nx.connected_components(G), key=len))

# Iterate over every edge
# Determine if edge is a strong or weak tie
edgeSet = nx.edges(G).keys()
for edge in edgeSet:
    n1 = edge[0]
    n2 = edge[1]
    if not haveStrongTie(G, n1, n2):
        G.remove_edge(n1,n2)

newCC = nx.number_connected_components(G)
maxNewCC = len(max(nx.connected_components(G), key=len))

print("Number of Connected Components originally: %d" % origCC)
print("Number of Connected Components without weak ties: %d" % newCC)
print("The number of nodes in the largest connected component originally: %d" % maxOrigCC)
print("The number of nodes in the largest connected component without weak ties: %d" % maxNewCC)
