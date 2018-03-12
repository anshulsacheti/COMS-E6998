#!/usr/local/bin/python3
import networkx as nx
import numpy as np
import pandas as pd
import dateutil.parser
import pdb
from datetime import datetime
import progressbar

# Define the coordinates of label A as anything within a radius of 10 miles
# define New York as (40.730610, -73.935242), Rio de Janeiro as (-22.970722, - 43.182365),
# and London as (51.509865, -0.118092).

# Read in graph
def generateGraph():
    """
    Generate edge graph via networkx and edge file

    Input:
        None

    Output:
        Networkx graph with all user interaction edges

    Returns
        G: nx edge graph
    """

    G = nx.read_edgelist(path="edges_train_anon.txt", nodetype=int)
    G = G.to_undirected()
    return G

def readCheckinData(G):
    """
    Reads checkin data file and stores it to a pandas df

    Input:
        G: networkx graph

    Output: pandas df with all checkin data

    Returns:
        checkinDF: pandas df
    """

    checkinDF = pd.read_csv("checkins_train_anon.txt", delimiter="\t", header=None, index_col=False,
                            names=["nodeNum", "dt", "longitude", "latitude"])
    checkinDF['dt'] = pd.to_datetime(checkinDF['dt'])

    graphNodes = np.array(list(G.nodes))
    checkinDF = checkinDF[checkinDF['nodeNum'].isin(graphNodes)]
    checkinDF = checkinDF.loc[(checkinDF.longitude!=0) & (checkinDF.latitude!=0)]

    return checkinDF

def getPossibleSeedNodes(G, df, longitude, latitude, r):
    """
    Return all nodes in dataframe that are within Radius r of longLat

    Input:
        G: networkx graph
        df: Pandas df
        longitude: float
        latitude: float
        r: Radius in miles

    Output:
        seedList: ndarry of all nodes within Radius r of longitude and latitude

    Returns:
        seedList - 1d np array
    """

    d = convertMilesToDegrees(r)

    # Calculate distance between each checkin and reference point
    # If within radius d return row
    seedNodes = df.loc[np.sqrt(np.power(df.longitude-longitude,2) + np.power(df.latitude-latitude,2))<=d]
    seedList = seedNodes.nodeNum.unique()
    #
    # graphNodes = np.array(list(G.nodes))
    # seedList = np.intersect1d(localSeeds, graphNodes)
    return seedList

def generateCheckinVariance(df, count):
    """
    Calculates and returns nodes in descending order of variance in check in location with at least count checkins

    Inputs:
        df: pandas dataframe
        count: integer

    Outputs:
        nodes: np 1d array

    Returns nodes in descending order of variance
    """
    def calculate_llvar(row):
        return np.sqrt(np.power(row.longitude,2)+np.power(row.latitude,2))

    g = df.groupby(['nodeNum'], sort=False)
    g = g.filter(lambda x: len(x) > count)
    g = g.groupby(['nodeNum'], sort=False).var()
    g.loc[:,'llvar'] = g.apply(calculate_llvar, axis=1)
    nodes = list(g.sort_values('llvar', ascending=False).index)
    return nodes

def generateCheckinMeanDict(df):
    """
    Calculates and returns dictionary of mean check in location for each node

    Inputs:
        df: pandas dataframe

    Outputs:
        nodes: dict

    Returns node dictionary with mean long/lat of checkin locations
    """

    g = df.groupby(['nodeNum'], sort=False).mean()
    nodes = g.to_dict('index')

    return nodes

def removeNodesWithNeighborsLessThanN(G, N):
    """
    Calculates and returns nodes with strictly more than N neighbors

    Inputs:
        G: networkx graph
        N: integer

    Outputs:
        nodes: np 1d array

    Returns nodes with more than N neighbors
    """

    graphNodes = np.array(list(G.nodes))
    nodes = []
    for n in graphNodes:
        if len(list(G.neighbors(n))) > N:
            nodes.append(n)

    return nodes

def avgUserCheckinsPerUserLocation(df, r):
    """
    Calculate average number of users who had checkins at a location where a user had a checkin

    Inputs:
        df: pandas dataframe
        r: radius r in miles

    Outputs:
        nodes: dict

    Returns node dictionary with mean number of users at each checkin location for some user
    """

    graphNodes = df.nodeNum.unique()
    d = convertMilesToDegrees(r)
    nodes = {}
    # nodes = dict(zip(graphNodes,np.zeros(len(graphNodes))))

    # Compare long/lat each node checkin with other nodes
    # Count average number of similar check in locations for that node
    bar = progressbar.ProgressBar()

    for gN in graphNodes:
        subDF = df.loc[df.nodeNum==gN]

        # Remove locations near one another to help runtime concerns
        i = 0
        while i < subDF.shape[0]:
            index = subDF.index[i]
            longitude = subDF.loc[index].longitude
            latitude = subDF.loc[index].latitude
            # Might want to scale by number of rows deleted, also might want to loop this across all rows
            subDF = subDF.loc[(subDF.index!=index) & (np.sqrt(np.power(subDF.longitude-longitude,2) + np.power(subDF.latitude-latitude,2))<=d)]

            i += 1

        dfDict = subDF[["longitude","latitude"]].to_dict('index')

        total = 0
        keys = list(dfDict.keys())
        for keyIdx in range(len(keys)):
            key = keys[keyIdx]
            longitude = dfDict[key]["longitude"]
            latitude = dfDict[key]["latitude"]
            total += (df.loc[(df.nodeNum!=gN) & (np.sqrt(np.power(df.longitude-longitude,2) + np.power(df.latitude-latitude,2))<=d)]).shape[0]

        # Handle div by 0
        if len(list(dfDict.keys()))!=0:
            total /= len(list(dfDict.keys()))

        # Handle div by 0
        if total == 0:
            total = 1
        nodes[gN] = total

    return nodes

        # Get all other checkins near same location as her checkin


def markNodeSetA(G, setA):
    """
    Adds attribute to all A nodes to handle them correctly for promotion

    Input:
        G: networkx graph
        setA: list of nodes

    Returns updated graph G with A nodes marked
    """

    nodeAdict = dict(zip(setA, [1]*len(setA)))
    nx.set_node_attributes(G, name="isA", values=nodeAdict)
    return G

def convertMilesToDegrees(miles):
    """
    Converts miles to degrees
    1 degree in longitudinal coordinates is approximately 55 miles

    Input:
        Miles: Integer/float

    Output:
        Degrees: Integer/float

    Returns number of degrees
    """

    ratioD2M = 1./55
    degrees = ratioD2M*miles
    return degrees
