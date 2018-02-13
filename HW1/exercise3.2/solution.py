import networkx as nx

# Read in edges
G = nx.read_edgelist("./CA-GrQc.txt")

# Get degree histogram
degree_hist = nx.degree_histogram(G)

TotalAuthors = float(sum(degree_hist))

# 1 co author
print("Percent of authors with 1 coauthor: %f" % (sum(degree_hist[1:2])/TotalAuthors))

# 10 or fewer
print("Percent of authors with 10 coauthor: %f" % (sum(degree_hist[0:11])/TotalAuthors))

# 20 or fewer
print("Percent of authors with 20 coauthor: %f" % (sum(degree_hist[0:21])/TotalAuthors))

# 40 or fewer
print("Percent of authors with 40 coauthor: %f" % (sum(degree_hist[0:41])/TotalAuthors))

# 80 or fewer
print("Percent of authors with 80 coauthor: %f" % (sum(degree_hist[0:81])/TotalAuthors))
