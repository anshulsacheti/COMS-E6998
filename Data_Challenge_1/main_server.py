#!/usr/local/bin/python3
import numpy as np
import pandas as pd
import dateutil.parser
import argparse
import sys
import pdb
import time
import requests
import json

# Class to manage server calls
class serverCalls:
    def __init__(self):
        headers = {'uni': 'as5105', 'pass': '21'}
        self.key = requests.get("http://167.99.225.109:5000/api/getKey", headers=headers).text
        self.callCount = 0
        self.api_data = {}
        self.lastRandomNode = []
        self.lastGetNodeInfo = []

    def getRandomNode(self):
        headers = {"uni": "as5105", "key":self.key}
        if self.callCount>=500:
            return self.lastRandomNode
        else:
            try:
                nodeInfo = requests.get("http://167.99.225.109:5000/api/nodes/getRandomNode", headers=headers)
                node = nodeInfo.json()
            except json.decoder.JSONDecodeError:
                with open('tmp.json', 'w') as output:
                    json.dump(self.api_data, output)
                print("Bad node query: %s" % nodeInfo)
                self.callCount += 1
                return self.lastRandomNode

            self.callCount += 1
            self.api_data[node["nodeid"]] = node
            self.lastRandomNode = node

            with open('tmp.json', 'a') as output:
                json.dump(node, output)

        # Self destruct
        self.destruct()

        return node

    def getNodeInfo(self, node):
        headers = {"uni": "as5105", "key":self.key}
        if self.callCount>=500:
            return self.lastGetNodeInfo
        else:
            if str(node) in list(self.api_data.keys()):
                node = self.api_data[str(node)]
            else:
                try:
                    nodeInfo = requests.get("http://167.99.225.109:5000/api/nodes/" + str(node) + "/NodeInfo", headers=headers)
                    node = nodeInfo.json()
                except json.decoder.JSONDecodeError:
                    with open('tmp.json', 'w') as output:
                        json.dump(self.api_data, output)
                    print("Bad node query: %s" % nodeInfo)
                    self.callCount += 1
                    return self.lastGetNodeInfo

                self.api_data[node["nodeid"]] = node
                self.callCount += 1
                self.lastGetNodeInfo = node

                with open('tmp.json', 'a') as output:
                    json.dump(node, output)

        # Self destruct
        self.destruct()

        return node

    # Write out all data if we go over call limit to make sure we have data
    def destruct(self):
        if self.callCount >= 499:
            with open('tmp.json', 'w') as output:
                json.dump(self.api_data, output)

            # print("Call count: %d, %s" % (self.callCount, self.api_data))

    def callsAvailable(self):
        return self.callCount < 500


def findSeedSet(seedSize, conversionRate, runType):
    """
    Calculate seed set given local subset with all their checkin info and other related data

    Input:
        checkinDF: pandas df with all checkin data
        G: networkx graph
        budget: int
        conversionRate: float
        longitude: float
        latitude: float
        runType: str
        checkins: int
        neighborMinimum: int

    Output:
        seedListA = list
        seedListB = list

    Returns:
        List of seeds that will best generate profit given current network
    """

    G = serverCalls()

    seedSet = []

    if runType=="smartWalk":
        # Much better score if A nodes hit, make sure 1 in set
        haveA = False

        # For a given node, decide if more beneficial to use it's neighbors or get new node
        # Metric for evaluating value of a given node
        seedNode = G.getRandomNode()
        seedSet = [seedNode["nodeid"]]

        i = 0
        while i<(seedSize-1+haveA) and G.callsAvailable():

            nodeNotAdded = True

            count = 1
            while nodeNotAdded and G.callsAvailable():
                # Walk
                for j in range(5):
                    neighbors = [k for k in seedNode["neighbors"].keys()]
                    n = np.random.choice(neighbors)
                    seedNode = G.getNodeInfo(n)
                    try:
                        clusteringCoef = float(seedNode["clusteringCoef"])
                    except ValueError:
                        clusteringCoef = 0

                # Check if node unique
                if seedNode["nodeid"] not in seedSet and clusteringCoef>0.01:
                    nodeNotAdded = False
                    seedSet.append(seedNode["nodeid"])
                    seedNode = G.getNodeInfo(seedNode["nodeid"])
                    if seedNode["label"]=='A':
                        haveA = True
                    continue

                # Got into loop
                count -= 1
                if count == 0:
                    count = 1
                    seedNode = G.getRandomNode()
                    while seedNode["nodeid"] in seedSet and G.callsAvailable():
                        seedNode = G.getRandomNode()

            i += 1

        # Add label=A node
        if not haveA:
            foundA = False
            queryData = G.api_data
            queryDataKeys = [key for key in queryData.keys()]
            for nodeID in queryDataKeys:
                if queryData[nodeID]["label"] =='A' and str(nodeID) not in seedSet:
                    seedSet.append(nodeID)
                    break
                # Go over all neighbors too
                neighbors = [k for k in queryData[nodeID]["neighbors"].keys()]
                for n in neighbors:
                    if queryData[nodeID]["neighbors"][n]["label"] =='A' and str(n) not in seedSet:
                        seedSet.append(n)
                        foundA = True
                        break

                # Handle above for loop finding value
                if foundA:
                    break

        # Pad output with random nodes (either from randomly generated seeds or from previously queried nodes)
        while len(seedSet)<seedSize:
            if G.callsAvailable():
                node = G.getRandomNode()["nodeid"]
                if node not in seedSet:
                    seedSet.append(node)
            else:
                queryData = G.api_data
                queryDataKeys = [key for key in queryData.keys()]
                node = np.random.choice(queryDataKeys)
                # Add main node
                if node not in seedSet:
                    seedSet.append(node)
                else:
                    # Try adding a neighbor
                    neighbors = [k for k in queryData[node]["neighbors"].keys()]
                    n = np.random.choice(neighbors)
                    if n not in seedSet:
                        seedSet.append(n)

    # Random Walk of various distances
    if runType=="randomWalk":

        seedNode = G.getRandomNode()
        seedSet = [seedNode["nodeid"]]

        for i in range(seedSize-1):

            nodeNotAdded = True

            count = 1
            while nodeNotAdded:
                # Walk
                for j in range(2):
                    neighbors = [k for k in seedNode["neighbors"].keys()]
                    n = np.random.choice(neighbors)
                    seedNode = G.getNodeInfo(n)

                # Check if node unique
                if seedNode["nodeid"] not in seedSet:
                    nodeNotAdded = False
                    seedSet.append(seedNode["nodeid"])
                    seedNode = G.getNodeInfo(seedNode["nodeid"])
                    continue

                # Got into loop
                count -= 1
                if count == 0:
                    count = 1
                    seedNode = G.getRandomNode()
                    while seedNode["nodeid"] in seedSet:
                        seedNode = G.getRandomNode()

    if runType=="ClusteringCoef":
        while len(seedSet)<seedSize:
            node = G.getRandomNode()
            nodeid = node["nodeid"]
            try:
                clusteringCoef = float(node["clusteringCoef"])
            except ValueError:
                clusteringCoef = 0
            if nodeid not in seedSet and clusteringCoef>0.0001:
                seedSet.append(nodeid)
    # Randomized
    if runType=="Randomized":
        while len(seedSet)<seedSize:
            node = G.getRandomNode()["nodeid"]
            if node not in seedSet:
                seedSet.append(node)

    # Convert final seedSet to A/B nodes
    seedsA = seedSet
    seedsB = []

    print("Number of calls by server: %d, nodes in set: %d" % (G.callCount, len(seedSet)))
    return [seedsA, seedsB]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--budget", type=int)
    parser.add_argument("--p", type=float)
    # parser.add_argument("--seedLocation", type=str)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--runType", type=str)
    # parser.add_argument("--checkins", type=int)
    # parser.add_argument("--neighborMin", type=int)

    args = parser.parse_args()

    # inputs
    seedSize = args.budget
    conversionRate = args.p
    iters = args.iters
    runType = args.runType
    # checkins = args.checkins
    # location = args.seedLocation
    #
    # if location.lower()=="new york":
    #     longitude   = 40.730610
    #     latitude    = -73.935242
    # elif location.lower()=="rio de janeiro":
    #     longitude   = -22.970722
    #     latitude    = -43.182365
    # elif location.lower()=="london":
    #     longitude   = 51.509865
    #     latitude    = -0.118092
    # elif location.lower()=="los angeles":
    #     # Correct values are -118, but using 118 as that is the correct reference
    #     longitude   = 34.0522
    #     latitude    = 118.2437
    # elif location.lower()=="san francisco":
    #     longitude   = -122.4194
    #     latitude    = 37.7749
    # else:
    #     print("Don't know how to map that location... EXITING")
    #     sys.exit()

    # # Initialize dataset
    # G = process_data.generateGraph()
    # checkinDF = process_data.readCheckinData(G)
    #
    # # Get seed set
    # seedList = process_data.getPossibleSeedNodes(G, checkinDF, longitude, latitude, 10)
    # G = process_data.markNodeSetA(G, seedList)

    # Set up loop values
    # totalProfit = 0.0
    # bestProfit = 0.0
    # profitList = np.array([])
    # bestSeedSet = []

    # Iterate
    print("RunType: %s" % runType)
    start = time.time()
    for i in range(iters):
        seedsA, seedsB = findSeedSet(seedSize, conversionRate, runType)
        print(seedsA)
        print(len(seedsA))
    #     profit, nodesWithProduct = simulate_graph.simulate(G, seedsA, seedsB, conversionRate)
    #     totalProfit += profit
    #     profitList = np.append(profitList, np.array([profit]))
    #
    #     # Store best case
    #     if profit>bestProfit:
    #         bestProfit = profit
    #         bestSeedSet = np.append(seedsA, seedsB)
    #         bestSeedSet = np.sort(bestSeedSet)
    #         print(bestSeedSet)
    #
    #     # Calculate percentage of A
    #     nodeAcount = 0
    #     for n in nodesWithProduct:
    #         # Node in set A
    #         if G.node[int(n)]!={}:
    #             nodeAcount += 1
    #
    #     try:
    #         percentNodesWithA = nodeAcount / len(seedList)
    #     except ZeroDivisionError:
    #         percentNodesWithA = 0
    #
    #     if i % (iters/100) == 0:
    #         print("On iter %d, " % (i), end='')
    #         print("Best profit: %d" % (bestProfit), end='')
    #         print(", Unique nodes in set: %d" % len(list(set(bestSeedSet))), end='')
    #         print(", Percent Nodes with A: %.2f" % percentNodesWithA, end='')
    #         if i>0:
    #             print(", Current avg: %d" % int(totalProfit*1.0/(i+1)), end ='')
    #         print("\n------------")
    #
    # print("Run time: %f" % (time.time()-start))
    # print("Profit over %d iters: %.1f" % (iters, totalProfit*1.0/iters))
    # print("Best profit: %d" % (bestProfit))
    #
    # # hist, bins = np.histogram(a=profitList, bins="auto")
    # # print("Histogram Percent Values: %s" % (hist))
    # # print("Histogram Bins: %s" % (bins))
    #
    # print("With seedset:\n%s" % (bestSeedSet))
