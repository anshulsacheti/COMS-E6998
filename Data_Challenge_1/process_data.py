#!/usr/local/bin/python3
import networkx as nx
import numpy as np
import pandas as pd
import dateutil.parser
import pdb

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

    return checkinDF
    # with open("checkins_train_anon.txt") as f:
    #     for line in f:
    #         data=line.split()
    #
    #         nodeNum = int(data[0])
    #         checkinDateTime = dateutil.parser.parse(data[1])
    #         longitude = float(data[2])
    #         latitude = float(data[3])
    #         _ = data [4] # data hash
    #
    #         # dateutil.parser.parse(checkinDateTime)
    #
    #         checkinDict = {}
    #         if nodeDict[nodeNum]!={}:
    #             nodeDict[nodeNum] = {}
    #         else:
    #             nodeDict[nodeNum][checkinDateTime] = {"longitude": longitude, "latitude": latitude}
    #
    #         nodeDict[nodeNum][]
    #         print(x)
    #         break

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
    Calculates and returns nodes in descending order of variance in check in location

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
