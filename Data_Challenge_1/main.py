#!/usr/local/bin/python3
import networkx as nx
import numpy as np
import pandas as pd
import dateutil.parser
import argparse
import sys
import simulate_graph
import process_data

def findSeedSet(checkinDF, localSeeds, G, budget, conversionRate, longitude, latitude):
    """
    Calculate seed set given local subset with all their checkin info and other related data

    Input:
        checkinDF: pandas df with all checkin data
        localSeeds: list
        G: networkx graph
        budget: int
        conversionRate: float
        longitude: float
        latitude: float

    Output:
        seedListA = list
        seedListB = list

    Returns:
        List of seeds that will best generate profit given current network
    """

    # seedNodeEntries = df[df.nodeNum.isin(seedList)]
    # seedsA = []
    # seedsB = np.random.choice(checkinDF.nodeNum.unique(), budget)
    if 1==1:
        if len(localSeeds)<budget:
            seedsA = localSeeds

            # Be smart about adding unique nodes
            seedsB = []
            graphNodes = np.array(list(G.nodes))
            while len(seedsB) < (budget-len(localSeeds)):
                tmp = np.random.choice(graphNodes)
                if tmp not in seedsB:
                    seedsB.append(tmp)
        else:
            seedsA = np.random.choice(localSeeds, budget, replace=False)
            seedsB = []
    return [seedsA, seedsB]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--budget", type=int)
    parser.add_argument("--p", type=float)
    parser.add_argument("--seedLocation", type=str)

    args = parser.parse_args()

    seedSize = args.budget
    location = args.seedLocation
    conversionRate = args.p

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
        longitude   = 34.0522
        latitude    = -118.2437
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
    bestSeedSet = []
    iters = 1000

    # Iterate
    for i in range(iters):
        seedsA, seedsB = findSeedSet(checkinDF, seedList, G, seedSize, conversionRate, longitude, latitude)
        profit = simulate_graph.simulate(G, seedsA, seedsB, conversionRate)
        totalProfit += profit

        # Store best case
        if profit>bestProfit:
            bestProfit = profit
            bestSeedSet = np.append(seedsA, seedsB)


    print("Profit over %d iters: %d" % (iters, totalProfit*1.0/iters))
    print("Best profit: %d, with seedset:\n%s" % (bestProfit, bestSeedSet))
