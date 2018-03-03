#!/usr/local/bin/python3
import networkx as nx
import numpy as np
import pandas as pd
import dateutil.parser
import argparse
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

    seedsA = np.random.choice(localSeeds,budget, replace=False)
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

    # define New York as (40.730610, -73.935242), Rio de Janeiro as (-22.970722, - 43.182365),
    # and London as (51.509865, -0.118092).

    if location.lower()=="new york":
        longitude   = 40.730610
        latitude    = -73.935242
    elif location.lower()=="rio de janeiro":
        longitude   = -22.970722
        latitude    = -43.182365
    elif location.lower()=="london":
        longitude   = 51.509865
        latitude    = -0.118092
    else:
        printf("Don't know how to map that location... EXITING")

    G = process_data.generateGraph()
    checkinDF = process_data.readCheckinData()

    seedList = process_data.getPossibleSeedNodes(G, checkinDF, longitude, latitude, 10)

    seedsA, seedsB = findSeedSet(checkinDF, seedList, G, seedSize, conversionRate, longitude, latitude)
    profit = simulate_graph.simulate(G, seedsA, seedsB, conversionRate)

    print("Profit %d" % profit)
