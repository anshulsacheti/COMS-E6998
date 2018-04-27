x = []
with open("sx-stackoverflow-a2q.txt", "r") as file:
    x=file.read().splitlines()

nodeTimestamp = {}
for line in x:
    node1, node2, timestamp = line.split(" ")
    node1 = int(node1)
    node2 = int(node2)
    timestamp = int(timestamp)

    if node1 in nodeTimestamp:
        nodeTimestamp[node1] = min(nodeTimestamp[node1], timestamp)
    else:
        nodeTimestamp[node1] = timestamp

    if node2 in nodeTimestamp:
        nodeTimestamp[node2] = min(nodeTimestamp[node2], timestamp)
    else:
        nodeTimestamp[node2] = timestamp

with open("nodeTimestampCategory.txt", "w") as file:
    for key in nodeTimestamp:
        if nodeTimestamp[key] < 1380000000:
            category = "B"
        else:
            category = "A"
        file.write("{},{}\n".format(key,category))
