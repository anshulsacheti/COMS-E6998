#!/usr/local/bin/python3
import networkx as nx
import numpy as np
import pandas as pd
import dateutil.parser
import argparse
import sys
import simulate_graph
import process_data
import pdb
from multiprocessing import Pool
import dill
import os
import itertools
import progressbar
import time
import requests
import json
import pickle

# Class to manage server calls
class serverCalls:
    def __init__(self):
        headers = {"uni": "as5105"}
        self.key = requests.get("http://159.89.95.64:5000/api/getKey", headers=headers).text
        self.callCount = 0
        self.api_data = []

    def getRandomNode(self):
        headers = {"uni": "as5105", "key":self.key}
        node = requests.get("http://159.89.95.64:5000/api/nodes/getRandomNode", headers=headers).json()
        self.callCount += 1
        self.api_data.append(node)

        # Self destruct
        self.destruct()

        return node
    def getNodeInfo(self, node):
        headers = {"uni": "as5105", "key":self.key}
        nodeInfo = requests.get("http://159.89.95.64:5000/api/nodes/" + str(node) + "/NodeInfo", headers=headers).json()
        self.api_data.append(nodeInfo)
        self.callCount += 1

        # Self destruct
        self.destruct()

        return nodeInfo

    def destruct(self):
        if self.callCount >=500:
            with open('tmp.txt', 'w') as output:
                output.write('%s\n' % json.dumps(Pickle.dumps(self.api_data)))

            # print("Call count: %d, %s" % (self.callCount, self.api_data))

    def callsAvailable(self):
        return self.callCount < 500


def findSeedSet(seedSize, conversionRate, runType, checkins):
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
        while G.callsAvailable():
            break

        # Pad output with random nodes (either from randomly generated seeds or from previously queried nodes)
        while len(seedSet)<seedSize:
            seedSet.append(SOME_VAL)

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
            if nodeid not in seedSet and clusteringCoef>0.01:
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
    parser.add_argument("--seedLocation", type=str)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--runType", type=str)
    parser.add_argument("--checkins", type=int)
    parser.add_argument("--neighborMin", type=int)

    args = parser.parse_args()

    # inputs
    seedSize = args.budget
    conversionRate = args.p
    iters = args.iters
    runType = args.runType
    checkins = args.checkins
    location = args.seedLocation

    if location.lower()=="new york":
        longitude   = 40.730610
        latitude    = -73.935242
    elif location.lower()=="rio de janeiro":
        longitude   = -22.970722
        latitude    = -43.182365
    elif location.lower()=="london":
        longitude   = 51.509865
        latitude    = -0.118092
    elif location.lower()=="los angeles":
        # Correct values are -118, but using 118 as that is the correct reference
        longitude   = 34.0522
        latitude    = 118.2437
    elif location.lower()=="san francisco":
        longitude   = -122.4194
        latitude    = 37.7749
    else:
        print("Don't know how to map that location... EXITING")
        sys.exit()

    # Initialize dataset
    G = process_data.generateGraph()
    checkinDF = process_data.readCheckinData(G)

    # Get seed set
    seedList = process_data.getPossibleSeedNodes(G, checkinDF, longitude, latitude, 10)
    G = process_data.markNodeSetA(G, seedList)

    # Set up loop values
    totalProfit = 0.0
    bestProfit = 0.0
    profitList = np.array([])
    bestSeedSet = []

    # Iterate
    print("RunType: %s" % runType)
    start = time.time()
    for i in range(iters):
        seedsA, seedsB = findSeedSet(seedSize, conversionRate, runType, checkins)
        profit, nodesWithProduct = simulate_graph.simulate(G, seedsA, seedsB, conversionRate)
        totalProfit += profit
        profitList = np.append(profitList, np.array([profit]))

        if len(seedList)>0:
            nodeAcount = 0
            for n in nodesWithProduct:
                # Node in set A
                if G.node[int(n)]!={}:
                    nodeAcount += 1

            percentNodesWithA = nodeAcount / len(seedList)

        # Store best case
        if profit>bestProfit:
            bestProfit = profit
            bestSeedSet = np.append(seedsA, seedsB)
            bestSeedSet = np.sort(bestSeedSet)
            print(bestSeedSet)

        if i % (iters/20) == 0:
            print("On iter %d, " % (i), end='')
            print("Best profit: %d" % (bestProfit), end='')
            if len(seedList)>0:
                print(", Percent Nodes with A: %.2f" % percentNodesWithA, end='')
            if i>0:
                print(", Current avg: %d" % int(totalProfit*1.0/(i+1)), end ='')
            print("\n------------")

    print("Run time: %f" % (time.time()-start))
    print("Profit over %d iters: %.1f" % (iters, totalProfit*1.0/iters))
    print("Best profit: %d" % (bestProfit))

    # hist, bins = np.histogram(a=profitList, bins="auto")
    # print("Histogram Percent Values: %s" % (hist))
    # print("Histogram Bins: %s" % (bins))

    print("With seedset:\n%s" % (bestSeedSet))
