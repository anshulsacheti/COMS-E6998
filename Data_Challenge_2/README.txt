Dataset: Stack Overflow Temporal Network
https://snap.stanford.edu/data/sx-stackoverflow.html

Dataset information
This is a temporal network of interactions on the stack exchange web site Stack Overflow.
The interaction represented by a directed edge (u, v, t):
user u answered user v's question at time t (in the graph sx-stackoverflow-a2q)
These graphs were constructed from the Stack Exchange Data Dump.
Node ID numbers correspond to the 'OwnerUserId' tag in that data dump.

The timestamps were categorized to above or below 1380000000 (chosen by finding a value
that most evenly split the dataset in 2) represented with an attribute "A"/"B" roughly giving
a 52%/48% split in total data. I included my code to do so.

Due to the size of the dataset, it was reduced to 400000 edges for reduced runtime (on the order of
an hour per iteration). This reduced and converted dataset is included with the submission. The edges
and attributes are separated into 2 files, similar to the first part of this Data Challenge.

Subset node count: 82436
Subset edge count: 400000

For testing:
I randomly take 100000 edges and remove them from the input dataset and compare after running how
my new graph of recommended edges compare in both accuracy and fairness. This is the case for testing
both the Instagram and Stackoverflow datasets.

How to run:

For Instagram Dataset:
Adamic Adar: python3 main.py --test --iters 30
Custom Algo: python3 main.py --test --iters 30 --SO

For Stackoverflow Dataset:
Adamic Adar: python3 main.py --test --iters 30 --SO
Custom Algo: python3 main.py --customAlgo --test --iters 30 --SO

Particulars of solution:

Adamic Adar:
I made a subgraph of all recommended edges. Thus if an edge is recommended multiple times, that
recommendation count is treated as 1. For an edge (u,v) since "v is recommended for u", I counted that
as an increment for node v's recommendation count, but nothing for node u. If there was a match in score,
then a node was randomly chosen and if a node had no neighbors then it's prediction was randomly chosen
from all nodes.

Custom Algo:
I built on the idea of common neighbors providing information about recommended edges. The algorithm
was similar to the skeleton provided by Adamic Adar. You generate a score based on neighbors and then
assign every relevant node a score, choosing the highest score as recommendation.

The algorithm first gets a list of mutual neighbors between node u and node v. For each of those mutual
neighbor nodes (let's call it the set X), you take x's neighbors. And for each of x's neighbors, increment
by 1 if x's neighbor is in the set of all nodes that could share a neighbor with u. Lastly, to encourage nodes
with low neighbor counts, we use a 1/np.log(sum) to strengthen nodes with fewer shared neighbors. This follows
a method very similar to Adamic Adar but is more akin to how strong a node v is as part of a clique with node u.

Similar to Adamic Adar, I made a subgraph of all recommended edges. Thus if an edge is recommended
multiple times, that recommendation count is treated as 1. For an edge (u,v) since "v is recommended
for u", I counted that as an increment for node v's recommendation count, but nothing for node u.
If there was a match in score, then a node was randomly chosen and if a node had no neighbors then
it's prediction was randomly chosen from all nodes.

Instagram Dataset (30 iters)
Custom:
Average Fairness: 0.066574
Average Percent correct: 0.000101

AA:
Average Fairness: 0.070224
Average Percent correct: 0.000094

Stackoverflow Dataset (30 iters)
Custom:
Average Fairness: 0.235560
Average Percent correct: 0.001138

AA:
Average Fairness: 0.234870
Average Percent correct: 0.001131

Results:

The new algorithm beats Adamic Adar on both average fairness and average percent correct on the Instagram
data, while only doing slightly better in average correctness on the stackoverflow data. The disparity
could be due to a variety of factors but it could be due to how Instagram is inherently groups of friends
interacting and thus cliques are more likely to be formed. This could skew the results as there are many nodes
that could meet the criteria to be recommended. Stackoverflow is primarily meant as Q/A for programmers and
thus having groups of friends interact there is unlikely. This means that edges that form are more unique
and are more likely to indicate a unique relationship and accordingly, strength in that relationship.
