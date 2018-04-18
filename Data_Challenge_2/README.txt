How to run: python3 main.py

Final fairness score: 0.069336
Predicted nodes file: final_predicted_nodes.csv

Particulars of solution:

I made a subgraph of all recommended edges. Thus if an edge is recommended multiple times, that
recommendation count is treated as 1. For an edge (u,v) since "v is recommended for u", I counted that
as an increment for node v's recommendation count, but nothing for node u. If there was a match in score,
then a node was randomly chosen and if a node had no neighbors then it's prediction was randomly chosen
from all nodes.

Dataset: Stack Overflow Temporal Network
https://snap.stanford.edu/data/sx-stackoverflow.html

Dataset information
This is a temporal network of interactions on the stack exchange web site Stack Overflow.
The interaction represented by a directed edge (u, v, t):
user u answered user v's question at time t (in the graph sx-stackoverflow-a2q)
These graphs were constructed from the Stack Exchange Data Dump.
Node ID numbers correspond to the 'OwnerUserId' tag in that data dump.
