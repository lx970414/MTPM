import networkx as nx
import scipy as sp
import numpy as np
import math
import warnings


def similarities(graph, method, flag=False):
    if method == 'MTPM':
        return MTPM_index(graph, flag)
    if method == 'CN':
        return common_neighbors_index(graph, flag)
    if method == 'RA':
        return resource_allocation_index(graph, flag)
    if method == 'Jaccard':
        return jaccard_coefficient(graph, flag)
    if method == 'PA':
        return preferential_attachment_index(graph,flag)
    # if method == 'AA':
    #     return AA_index(graph,flag)
    # if method == 'LP':
    #     return local_path_index(graph,flag)
    # if method == 'RALP':
    #     return RALP_index(graph,flag)
    
    

def pair(x, y):
    if (x < y):
        return (x, y)
    else:
        return (y, x)
    
def MTPM_index(G, for_existing_edge=False):

    ebunch = nx.edges(G) if for_existing_edge else nx.non_edges(G)
    ebunch = [pair(u, v) for u, v in ebunch]
    sim_dict = {}                                                             
    degree_dic = {}
    for v in G.nodes():
        degree_dic[v] = nx.degree(G,v)
       
    for u, v in ebunch:
        s1 = 0                                                                                
        cn_nodes1 = list(nx.common_neighbors(G, u, v))
        for node1 in cn_nodes1:
            s1 += (1 / degree_dic[ node1])
            
        # paths with length 3
        s2 = 0                                           
        for w in nx.neighbors(G, u):
            if w != v:                                   
                x, y = pair(w, v)
                cn_nodes2 = list(nx.common_neighbors(G, x, y))                
                ss = 0                                  
                for node2 in cn_nodes2:
                    if node2 != u:                        
                        degree_sum = 1 / degree_dic[w] + 1 / degree_dic[ node2]
                        ss += degree_sum
                s2 += ss 
        # end for
        s = s1 + 0.01 * s2
        sim_dict[(u, v)] = s
        # end if
    # end for
    return sim_dict



def common_neighbors_index(G, for_existing_edge=False):
     
    
    sim_dict = {}    
    node_num = nx.number_of_nodes(G)
    for u in G.nodes():
        for w in nx.neighbors(G, u):
            for v in nx.neighbors(G, w):
                if u < v:
                    if (for_existing_edge and G.has_edge(u, v)) or (not for_existing_edge and not G.has_edge(u, v)):
                        if (u,v) in sim_dict:
                            sim_dict[(u,v)] += 1
                        else:
                            sim_dict[(u,v)] = 1

    return sim_dict


def preferential_attachment_index(G, for_existing_edge=False):
    
    ebunch = nx.edges(G) if for_existing_edge else nx.non_edges(G)
    ebunch = [pair(u, v) for u, v in ebunch]                                     
    sim_dict = {}   
    degree_list = {}
    for v in G.nodes():
        degree_list[v] = nx.degree(G,v)

    for u, v in ebunch:
        s = degree_list[u] * degree_list[v]
        if s > 0:
            sim_dict[(u, v)] = s
        # end if
    # end for

    return sim_dict



def jaccard_coefficient(G, for_existing_edge=False):
    
        
    sim_dict = {}   
    node_num = nx.number_of_nodes(G)
    for u in G.nodes():
        for w in nx.neighbors(G, u):
            for v in nx.neighbors(G, w):
                if u < v:
                    if (for_existing_edge and G.has_edge(u, v)) or (not for_existing_edge and not G.has_edge(u, v)):
                        if (u,v) in sim_dict:
                            sim_dict[(u,v)] += 1.0
                        else:
                            sim_dict[(u,v)] = 1.0

    degree_list = {}
    for v in G.nodes():
        degree_list[v] = nx.degree(G,v)

    for (u, v) in sim_dict.keys():
        s = sim_dict[(u, v)]
        sim_dict[(u, v)] = s / (degree_list[u] + degree_list[v] - s)
    
    
    return sim_dict




def resource_allocation_index(G, for_existing_edge=False):
    
    sim_dict = {}     
    node_num = nx.number_of_nodes(G)
    degree_list = {}
    for v in G.nodes():
        degree_list[v] = nx.degree(G,v)

    for u in G.nodes():
        for w in nx.neighbors(G, u):
            for v in nx.neighbors(G, w):
                if u < v:
                    if (for_existing_edge and G.has_edge(u, v)) or (not for_existing_edge and not G.has_edge(u, v)):
                        if (u, v) in sim_dict:
                            sim_dict[(u, v)] += 1 / degree_list[w]
                        else:
                            sim_dict[(u, v)] = 1 / degree_list[w]

    return sim_dict



