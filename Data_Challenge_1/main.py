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

class MP:
    @staticmethod
    def union(a):
        # print("a: %s" % (a))
        a0=a[0]
        a1=a[1]
        u = np.union1d(a0, a1)
        # print("u: %s" % (u))
        return u.size

def findSeedSet(checkinDF, G, budget, conversionRate, longitude, latitude, runType):
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

    Output:
        seedListA = list
        seedListB = list

    Returns:
        List of seeds that will best generate profit given current network
    """

    # seedNodeEntries = df[df.nodeNum.isin(seedList)]
    # seedsA = []
    # seedsB = np.random.choice(checkinDF.nodeNum.unique(), budget)

    mp = MP()
    bar = progressbar.ProgressBar()

    if runType == "checkinVariance":

        graphNodes = process_data.generateCheckinVariance(checkinDF, 5)[0:1000]
        seedSet = np.random.choice(graphNodes, budget, replace=False)

        # neighborSet = np.array([])
        # seedSet = []
        #
        # neighborsPerNode = []
        # for node in graphNodes:
        #     neighborsPerNode.append(list(G.neighbors(node)))
        #
        # # Much slower
        # # with Pool() as pool:
        # #     neighbors = pool.map(G.neighbors, graphNodes)
        # #     neighborsList = pool.map(list, neighbors)
        # #     neighborsExpanded = pool.map(np.array, neighborsList)
        #
        #
        # # Get nodes that add the most neighbors
        # for i in bar(range(budget)):
        #     mostNeighbors = 0
        #     bestNode = 0
        #     uselessNodes = []
        #     # Attempt at multiprocessing
        #     if 1==0:
        #         with Pool() as pool:
        #             results = pool.map(mp.union, list(zip([list(neighborSet)]*len(neighborsList), neighborsList)))
        #             # print("results: %s" % (results))
        #             bestNode = graphNodes[np.argmax(results)]
        #
        #     # Find node that adds the most unique neighbors
        #     else:
        #         for j,nodeSet in enumerate(neighborsPerNode):
        #             union = np.union1d(nodeSet, neighborSet)
        #
        #             # Get most new neighbors added
        #             if union.size > mostNeighbors:
        #                 mostNeighbors = union.size
        #
        #                 bestNodeLoc = j
        #                 bestNode = graphNodes[j]
        #
        #         neighborsPerNode = list(np.delete(neighborsPerNode,uselessNodes))
        #     seedSet.append(bestNode)
        #     neighborSet = np.union1d(list(G.neighbors(bestNode)), neighborSet)

    if runType=="UsersCities":
        nyc = process_data.getPossibleSeedNodes(G, checkinDF, 40.730610, -73.935242, 10)
        rdj = process_data.getPossibleSeedNodes(G, checkinDF, -22.970722, -43.182365, 10)
        ldn = process_data.getPossibleSeedNodes(G, checkinDF, 51.509865, -0.118092, 10)
        las = process_data.getPossibleSeedNodes(G, checkinDF, 34.0522, -118.2437, 10)

        graphNodes = np.append(las, np.append(ldn, np.append(nyc,rdj)))

        neighborSet = np.array([])
        seedSet = []

        neighborsPerNode = []
        for node in graphNodes:
            neighborsPerNode.append(list(G.neighbors(node)))

        # Much slower
        # with Pool() as pool:
        #     neighbors = pool.map(G.neighbors, graphNodes)
        #     neighborsList = pool.map(list, neighbors)
        #     neighborsExpanded = pool.map(np.array, neighborsList)


        # Get nodes that add the most neighbors
        for i in bar(range(budget)):
            mostNeighbors = 0
            bestNode = 0
            uselessNodes = []
            # Attempt at multiprocessing
            if 1==0:
                with Pool() as pool:
                    results = pool.map(mp.union, list(zip([list(neighborSet)]*len(neighborsList), neighborsList)))
                    # print("results: %s" % (results))
                    bestNode = graphNodes[np.argmax(results)]

            # Find node that adds the most unique neighbors
            else:
                for j,nodeSet in enumerate(neighborsPerNode):
                    union = np.union1d(nodeSet, neighborSet)

                    # Get most new neighbors added
                    if union.size > mostNeighbors:
                        mostNeighbors = union.size

                        bestNodeLoc = j
                        bestNode = graphNodes[j]

                neighborsPerNode = list(np.delete(neighborsPerNode,uselessNodes))
            seedSet.append(bestNode)
            neighborSet = np.union1d(list(G.neighbors(bestNode)), neighborSet)

    if runType=="GreedyLocation":
        pass

    # Greedy (adds most neighbors)
    if runType=="GreedyNeighbor":

        graphNodes = np.array(list(G.nodes))
        neighborSet = np.array([])
        seedSet = []

        neighborsPerNode = []
        for node in graphNodes:
            neighborsPerNode.append(list(G.neighbors(node)))

        # Much slower
        # with Pool() as pool:
        #     neighbors = pool.map(G.neighbors, graphNodes)
        #     neighborsList = pool.map(list, neighbors)
        #     neighborsExpanded = pool.map(np.array, neighborsList)


        # Get nodes that add the most neighbors
        for i in bar(range(budget)):
            mostNeighbors = 0
            bestNode = 0
            uselessNodes = []
            # Attempt at multiprocessing
            if 1==0:
                with Pool() as pool:
                    results = pool.map(mp.union, list(zip([list(neighborSet)]*len(neighborsList), neighborsList)))
                    # print("results: %s" % (results))
                    bestNode = graphNodes[np.argmax(results)]

            # Find node that adds the most unique neighbors
            else:
                for j,nodeSet in enumerate(neighborsPerNode):
                    union = np.union1d(nodeSet, neighborSet)

                    # Get most new neighbors added
                    if union.size > mostNeighbors:
                        mostNeighbors = union.size

                        bestNodeLoc = j
                        bestNode = graphNodes[j]

                neighborsPerNode = list(np.delete(neighborsPerNode,uselessNodes))
            seedSet.append(bestNode)
            neighborSet = np.union1d(list(G.neighbors(bestNode)), neighborSet)

    # Randomized
    if runType=="Randomized":
        graphNodes = np.array(list(G.nodes))
        seedSet = np.random.choice(graphNodes, budget, replace=False)

    # Convert final seedSet to A/B nodes
    seedsA = []
    seedsB = []
    for seed in seedSet:
        if G.node[seed]!={}:
            seedsA.append(seed)
        else:
            seedsB.append(seed)

    return [seedsA, seedsB]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--budget", type=int)
    parser.add_argument("--p", type=float)
    parser.add_argument("--seedLocation", type=str)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--runType", type=str)

    args = parser.parse_args()

    seedSize = args.budget
    location = args.seedLocation
    conversionRate = args.p
    iters = args.iters
    runType = args.runType

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
    else:
        print("Don't know how to map that location... EXITING")
        sys.exit()

    # Initialize dataset
    G = process_data.generateGraph()
    checkinDF = process_data.readCheckinData(G)

    # Get seed set
    seedList = process_data.getPossibleSeedNodes(G, checkinDF, longitude, latitude, 10)
    G = process_data.markNodeSetA(G, seedList)

    totalProfit = 0.0
    bestProfit = 0.0
    profitList = np.array([])
    bestSeedSet = []


    # Iterate
    print("RunType: %s" % runType)
    start = time.time()
    for i in range(iters):
        seedsA, seedsB = findSeedSet(checkinDF, G, seedSize, conversionRate, longitude, latitude, runType)
        profit = simulate_graph.simulate(G, seedsA, seedsB, conversionRate)
        totalProfit += profit
        profitList = np.append(profitList, np.array([profit]))

        # Store best case
        if profit>bestProfit:
            bestProfit = profit
            bestSeedSet = np.append(seedsA, seedsB)

        if i % (iters/10) == 0:
            print("On iter %d" % (i))
            print("Best profit: %d" % (bestProfit))
            if i>0:
                print("Current avg: %d" % int(totalProfit*1.0/(i+1)))
            print("------------")

    print("Run time: %f" % (time.time()-start))
    print("Profit over %d iters: %d" % (iters, totalProfit*1.0/iters))
    print("Best profit: %d" % (bestProfit))

    # hist, bins = np.histogram(a=profitList, bins="auto")
    # print("Histogram Percent Values: %s" % (hist))
    # print("Histogram Bins: %s" % (bins))

    print("With seedset:\n%s" % (bestSeedSet))
