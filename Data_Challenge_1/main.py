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

def findSeedSet(checkinDF, G, budget, conversionRate, longitude, latitude, runType, checkins, neighborMinimum, CheckinsPerLocation):
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

    # seedNodeEntries = df[df.nodeNum.isin(seedList)]
    # seedsA = []
    # seedsB = np.random.choice(checkinDF.nodeNum.unique(), budget)

    mp = MP()

    if runType=="CheckinsPerLocation":
        graphNodes  = list(G.nodes)

        # Calculate average number of users who had checkins at a location where a user had a checkin
        avgLocCheckinsByNode = CheckinsPerLocation
        checkinKeys = list(avgLocCheckinsByNode.keys())

        # Restrict set of nodes to only those in first connected cc (largest)
        cc = [G.subgraph(c) for c in nx.connected_components(G)]

        # Generate probability distribution for each node in a component
        ccNodeDistribution = []
        for comp in cc:
            nodes = np.intersect1d(list(comp), checkinKeys)
            invertedAvgCheckins = [np.reciprocal(avgLocCheckinsByNode[node]) for node in nodes]
            # np.reciprocal(list(avgLocCheckinsByNode.values()))
            sumInvertedAvgCheckins = sum(invertedAvgCheckins)
            checkinDistribution = [ci/sumInvertedAvgCheckins for ci in invertedAvgCheckins]
            ccNodeDistribution.append(checkinDistribution)

        # Generate probability of picking a component
        totalNodeCount = len(G.nodes)
        ccDistribution = [len(list(comp))/totalNodeCount for comp in cc]

        # minNeighborNodes = process_data.removeNodesWithNeighborsLessThanN(G,neighborMinimum)
        # graphNodes = np.intersect1d(minCheckinNodes, minNeighborNodes)

        seedSet = []

        while len(seedSet)<budget:

            # Pick component
            componentNodes = np.array([])
            while componentNodes.size==0:
                componentIdx = np.random.choice(range(len(cc)), 1, replace=True, p=ccDistribution)[0]
                componentNodes = np.array(list(cc[componentIdx]))
                # intersectNodes = np.intersect1d(graphNodes, componentNodes)

            try:
                node = np.random.choice(componentNodes, 1, replace=False, p=ccNodeDistribution[componentIdx])[0]
            except ValueError:
                pdb.set_trace()
            if node not in seedSet:
                seedSet.append(node)

    if "Components+Neighbors":
        graphNodes  = list(G.nodes)

        # Restrict set of nodes to only those in first connected cc (largest)
        cc = [G.subgraph(c) for c in nx.connected_components(G)]

        totalNodeCount = len(G.nodes)
        ccNodeDistribution = [len(list(comp))/totalNodeCount for comp in cc]

        # minNeighborNodes = process_data.removeNodesWithNeighborsLessThanN(G,neighborMinimum)
        # graphNodes = np.intersect1d(minCheckinNodes, minNeighborNodes)

        seedSet = []

        mostNeighbors = 0
        mostNeighborsFromA = 0

        # Pick A node with most neighbors
        nodeAlist = process_data.getPossibleSeedNodes(G, checkinDF, longitude, latitude, 10)
        if nodeAlist.size:
            for node in nodeAlist:
                neighbors = len(list(G.neighbors(node)))
                if neighbors > mostNeighbors:
                    mostNeighbors = neighbors
                    mostNeighborsFromA = node

            seedSet.append(mostNeighborsFromA)

        while len(seedSet)<budget:

            # Pick component
            componentNodes = np.array([])
            while componentNodes.size==0:
                componentIdx = np.random.choice(range(len(cc)), 1, replace=True, p=ccNodeDistribution)[0]
                componentNodes = np.array(list(cc[componentIdx]))
                # intersectNodes = np.intersect1d(graphNodes, componentNodes)

            # Not enough nodes left to make investment here worthwhile
            if (componentNodes.size - np.intersect1d(componentNodes,seedSet).size) > np.sqrt(componentNodes.size):
                node = np.random.choice(componentNodes, 1, replace=False)[0]
            else:
                continue

            if node not in seedSet:
                seedSet.append(node)

    if "Components+Sparsity":
        graphNodes  = list(G.nodes)

        # Restrict set of nodes to only those in first connected cc (largest)
        cc = [G.subgraph(c) for c in nx.connected_components(G)]

        totalNodeCount = len(G.nodes)
        ccNodeDistribution = [len(list(comp))/totalNodeCount for comp in cc]

        # minNeighborNodes = process_data.removeNodesWithNeighborsLessThanN(G,neighborMinimum)
        # graphNodes = np.intersect1d(minCheckinNodes, minNeighborNodes)

        nodeLocationDict = process_data.generateCheckinMeanDict(checkinDF)
        # varNodes  = process_data.generateCheckinVariance(checkinDF, checkins)[0:15000]

        componentChosenNodeLocations = [[] for x in range(len(cc))]
        seedSet = []

        # Get seed nodes
        while len(seedSet)<budget:

            # Pick component
            componentNodes = np.array([])
            while componentNodes.size==0:
                componentIdx = np.random.choice(range(len(cc)), 1, replace=True, p=ccNodeDistribution)[0]
                componentNodes = np.array(list(cc[componentIdx]))
                # componentNodes = np.intersect1d(varNodes, componentNodes)

            # Get node locations
            node = np.random.choice(componentNodes, 1, replace=False)[0]

            # Key might not have checkin data but could still be important
            try:
                nodeLL = nodeLocationDict[node]
            except KeyError:
                # Add only new nodes
                if node not in seedSet:
                    seedSet.append(node)
                    continue

            # Add only new nodes
            # Add nodes far away from other nodes in the same component
            if node not in seedSet:

                # Find nodes that are in unique locations
                # uniqueLoc = True
                ccLocations = np.array(componentChosenNodeLocations[componentIdx])
                if ccLocations.size>0:
                    uniqueLoc = np.any(np.sqrt(np.power(ccLocations[:,0]-nodeLL["longitude"],2)+np.power(ccLocations[:,1]-nodeLL["latitude"],2))<=process_data.convertMilesToDegrees(10))
                else:
                    uniqueLoc = True
                # for nodeLongitude, nodeLatitude in componentChosenNodeLocations[componentIdx]:
                #     if np.sqrt(np.power(nodeLL["longitude"]-nodeLongitude,2) + np.power(nodeLL["latitude"]-nodeLatitude,2))<=process_data.convertMilesToDegrees(0.0001):
                #         uniqueLoc = False
                #         break
                if uniqueLoc:
                    seedSet.append(node)
                    componentChosenNodeLocations[componentIdx].append([nodeLL["longitude"],nodeLL["latitude"]])

    if "Components":
        graphNodes  = list(G.nodes)

        # Restrict set of nodes to only those in first connected cc (largest)
        cc = [G.subgraph(c) for c in nx.connected_components(G)]

        totalNodeCount = len(G.nodes)
        ccNodeDistribution = [len(list(comp))/totalNodeCount for comp in cc]

        # minNeighborNodes = process_data.removeNodesWithNeighborsLessThanN(G,neighborMinimum)
        # graphNodes = np.intersect1d(minCheckinNodes, minNeighborNodes)

        seedSet = []

        while len(seedSet)<budget:

            # Pick component
            componentNodes = np.array([])
            while componentNodes.size==0:
                componentIdx = np.random.choice(range(len(cc)), 1, replace=True, p=ccNodeDistribution)[0]
                componentNodes = np.array(list(cc[componentIdx]))
                # intersectNodes = np.intersect1d(graphNodes, componentNodes)

            node = np.random.choice(componentNodes, 1, replace=False)[0]

            if node not in seedSet:
                seedSet.append(node)


    if runType=="NeighborsWithSparsity":
        graphNodes = process_data.removeNodesWithNeighborsLessThanN(G,neighborMinimum)

        nodeLocationDict = process_data.generateCheckinMeanDict(checkinDF)
        seedSet = []
        chosenNodeLocations = []

        while len(seedSet)<budget:

            node = np.random.choice(graphNodes, 1, replace=False)[0]
            while node not in nodeLocationDict.keys():
                node = np.random.choice(graphNodes, 1, replace=False)[0]
            nodeLL = nodeLocationDict[node]

            uniqueLoc = True
            for nodeLongitude, nodeLatitude in chosenNodeLocations:
                if np.sqrt(np.power(nodeLL["longitude"]-nodeLongitude,2) + np.power(nodeLL["latitude"]-nodeLatitude,2))<=process_data.convertMilesToDegrees(10):
                    uniqueLoc = False
                    break

            seedSet.append(node)
            chosenNodeLocations.append([nodeLL["longitude"],nodeLL["latitude"]])

    if runType=="Widespread+cc":
        graphNodes  = process_data.generateCheckinVariance(checkinDF, checkins)[0:2000]

        # Restrict set of nodes to only those in first connected cc (largest)
        cc = [G.subgraph(c) for c in nx.connected_components(G)]

        totalNodeCount = len(G.nodes)
        ccNodeDistribution = [len(list(comp))/totalNodeCount for comp in cc]

        # minNeighborNodes = process_data.removeNodesWithNeighborsLessThanN(G,neighborMinimum)
        # graphNodes = np.intersect1d(minCheckinNodes, minNeighborNodes)

        nodeLocationDict = process_data.generateCheckinMeanDict(checkinDF)
        seedSet = []
        chosenNodeLocations = []


        while len(seedSet)<budget:

            # Pick component
            intersectNodes = np.array([])
            while intersectNodes.size==0:
                componentIdx = np.random.choice(range(len(cc)), 1, replace=True, p=ccNodeDistribution)[0]
                componentNodes = list(cc[componentIdx])
                intersectNodes = np.intersect1d(graphNodes, componentNodes)

            node = np.random.choice(intersectNodes, 1, replace=False)[0]
            nodeLL = nodeLocationDict[node]

            # Find nodes that are in unique locations
            uniqueLoc = True
            for nodeLongitude, nodeLatitude in chosenNodeLocations:
                if np.sqrt(np.power(nodeLL["longitude"]-nodeLongitude,2) + np.power(nodeLL["latitude"]-nodeLatitude,2))<=process_data.convertMilesToDegrees(10):
                    uniqueLoc = False
                    break

            seedSet.append(node)
            chosenNodeLocations.append([nodeLL["longitude"],nodeLL["latitude"]])
        # seedSet = np.random.choice(graphNodes, budget, replace=False)


    if runType=="Widespread":
        graphNodes  = process_data.generateCheckinVariance(checkinDF, checkins)[0:2000]

        # minNeighborNodes = process_data.removeNodesWithNeighborsLessThanN(G,neighborMinimum)
        # graphNodes = np.intersect1d(minCheckinNodes, minNeighborNodes)

        nodeLocationDict = process_data.generateCheckinMeanDict(checkinDF)
        seedSet = []
        chosenNodeLocations = []


        while len(seedSet)<budget:

            node = np.random.choice(graphNodes, 1, replace=False)[0]
            nodeLL = nodeLocationDict[node]

            # Find nodes that are in unique locations
            uniqueLoc = True
            for nodeLongitude, nodeLatitude in chosenNodeLocations:
                if np.sqrt(np.power(nodeLL["longitude"]-nodeLongitude,2) + np.power(nodeLL["latitude"]-nodeLatitude,2))<=process_data.convertMilesToDegrees(5):
                    uniqueLoc = False
                    break

            seedSet.append(node)
            chosenNodeLocations.append([nodeLL["longitude"],nodeLL["latitude"]])
        # seedSet = np.random.choice(graphNodes, budget, replace=False)


    if runType=="GreedyRandomMostNeighbors":
        graphNodes = process_data.removeNodesWithNeighborsLessThanN(G,neighborMinimum)
        seedSet = np.random.choice(graphNodes, budget, replace=False)

    if runType=="CheckinVariance":

        graphNodes  = process_data.generateCheckinVariance(checkinDF, checkins)[0:2000]
        # minNeighborNodes = process_data.removeNodesWithNeighborsLessThanN(G,neighborMinimum)
        #
        # graphNodes = np.intersect1d(minCheckinNodes, minNeighborNodes)
        seedSet = np.random.choice(graphNodes, budget, replace=False)

    if runType=="UsersCities":
        bar = progressbar.ProgressBar()

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

    # Greedy (adds most neighbors)
    if runType=="GreedyNeighbor":
        bar = progressbar.ProgressBar()

        # graphNodes = np.array(list(G.nodes))

        graphNodes = process_data.removeNodesWithNeighborsLessThanN(G,neighborMinimum)
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

    # Get nodes with the most neighbors
    if runType=="Top100MostNeighbors":

        graphNodes = np.array(list(G.nodes))
        neighborSet = np.array([])
        seedSet = []

        neighborsPerNode = []
        for node in graphNodes:
            neighborsPerNode.append(list(G.neighbors(node)))

        # Get top 100
        neighborsPerNode.sort(key=len, reverse=True)
        nodesOrderedByNeighbor = sorted(range(len(neighborsPerNode)), key = lambda x: len(neighborsPerNode[x]), reverse=True)
        top100 = nodesOrderedByNeighbor[0:100]

        seedSet = []
        for n in top100:
            seedSet.append(graphNodes[n])

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
    parser.add_argument("--checkins", type=int)
    parser.add_argument("--neighborMin", type=int)

    args = parser.parse_args()

    seedSize = args.budget
    location = args.seedLocation
    conversionRate = args.p
    iters = args.iters
    runType = args.runType
    checkins = args.checkins
    neighborMinimum = args.neighborMin

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

    if runType == "CheckinsPerLocation":
        # Calculate average number of users who had checkins at a location where a user had a checkin
        avgLocCheckinsByNode = process_data.avgUserCheckinsPerUserLocation(checkinDF, 10)
    else:
        avgLocCheckinsByNode = {}

    totalProfit = 0.0
    bestProfit = 0.0
    profitList = np.array([])
    bestSeedSet = []


    # Iterate
    print("RunType: %s" % runType)
    start = time.time()
    for i in range(iters):
        seedsA, seedsB = findSeedSet(checkinDF, G, seedSize, conversionRate, longitude, latitude, runType, checkins, neighborMinimum, avgLocCheckinsByNode)
        profit = simulate_graph.simulate(G, seedsA, seedsB, conversionRate)
        totalProfit += profit
        profitList = np.append(profitList, np.array([profit]))

        # Store best case
        if profit>bestProfit:
            bestProfit = profit
            bestSeedSet = np.append(seedsA, seedsB)
            bestSeedSet = np.sort(bestSeedSet)

        if i % (iters/20) == 0:
            print("On iter %d, " % (i), end='')
            print("Best profit: %d" % (bestProfit), end='')
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
