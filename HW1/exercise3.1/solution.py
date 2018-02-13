import networkx as nx

# Read in edges
G = nx.read_edgelist("./CA-GrQc.txt")

# Get degree histogram
degree_hist = nx.degree_histogram(G)

# Find min degree
# Find first nonzero value in list
for i,val in enumerate(degree_hist):
    if val > 0:
        print("Min degree is %d" % i)
        break

# Find max degree
# Find last nonzero value in list
max_degree = -1
for i,val in enumerate(degree_hist):
    if val > 0:
        max_degree = i

print("Max degree is %d" % i)
