import networkx as nx
import numpy as np
import math
import random
import datetime
from DS import ds_sim_function

def pair(x, y):
    if (x < y):
        return (x, y)
    else:
        return (y, x)


def LinkPre(graph_file, out_file, sim_method, t, p, alpha,rele):

    G = nx.read_edgelist(graph_file + str(alpha) + '.edgelist', nodetype = int)   # G represents layer alpha of the multiplex network.

    node_num = nx.number_of_nodes(G)
    edge_num = nx.number_of_edges(G)

    non_edge_list = [pair(u, v) for u, v in nx.non_edges(G)]
    non_edge_num = len(non_edge_list)

    print("V: %d\tE: %d\tNon: %d" % (node_num, edge_num, non_edge_num))

    test_num = int(edge_num * p)
    pre_num = 0

    for l in range(10, 101, 10):
        if l < test_num:
            pre_num += 1
        else:
            break
        
    pre_num += 1

    print('test_edge_num: %d' % test_num)

    
    auc_list = []
    rs_list = []
    time_list = []
    pre_matrix = [[0 for it in range(t)] for num in range(pre_num)]        
    
    
    for it in range(t):
        if it % 10 == 0:
            print('turn: %d' % it)      
        

        seed = math.sqrt(edge_num * node_num) + math.pow((1 + it) * 10, 3)  
        random.seed(seed)
        rand_set = set(random.sample(range(edge_num), test_num))

        training_graph = nx.Graph()               
        training_graph.add_nodes_from(G.nodes())
        test_edge_list = []

        r = 0
        for u, v in nx.edges(G):
            u, v = pair(u, v)
            if r in rand_set:  
                test_edge_list.append((u, v))
            else:
                training_graph.add_edge(u, v)  
                
            r += 1
        training_graph.to_undirected()  
       

        start = datetime.datetime.now()
        
        sim_dict = ds_sim_function(training_graph, sim_method, alpha, rele, graph_file)   
        end = datetime.datetime.now()                                                 

        
        time_list.append((end - start).microseconds)
        
        
        auc_value = AUC(sim_dict, test_edge_list, non_edge_list)
        auc_list.append(auc_value)
       
        sim_list = [((u, v), s) for (u, v), s in sim_dict.items()]

        sim_dict.clear()

        sim_list.sort(key=lambda x: (x[1], x[0]), reverse=True)

        
        rank_score = Ranking_score(sim_list, test_edge_list, non_edge_num)
        rs_list.append(rank_score)   
        
        
        pre_list = Precision(sim_list, test_edge_list, test_num)

        for num in range(pre_num):           
            pre_matrix[num][it] = pre_list[num]
    

    
    auc_avg, auc_std = stats(auc_list)

    print('AUC: %.4f(%.4f)' % (auc_avg, auc_std))
    out_file.write('%.4f(%.4f)\t' % (auc_avg, auc_std))

    rs_avg, rs_std = stats(rs_list)

    print('Ranking_Score: %.4f(%.4f)' % (rs_avg, rs_std))
    out_file.write('%.4f(%.4f)\t' % (rs_avg, rs_std))

    time_avg, time_std = stats(time_list)

    print('Time: %.4f(%.4f)' % (time_avg, time_std))
    out_file.write('%.4f(%.4f)\t' % (time_avg, time_std))

    pre_avg_list = []
    pre_std_list = []
    for num in range(pre_num):
        pre_avg, pre_std = stats(pre_matrix[num])   
        pre_avg_list.append(pre_avg)
        pre_std_list.append(pre_std)
    

    print('Precision: ')
    for num in range(pre_num):
        print('%.4f(%.4f)\t' % (pre_avg_list[num], pre_std_list[num]))          
        out_file.write('%.4f(%.4f)\t' % (pre_avg_list[num], pre_std_list[num]))
    
    out_file.write('%d\n' % test_num)


def stats(value_list):
    value_array = np.array(value_list)
    avg = np.mean(value_array)
    std = np.std(value_array)

    return avg, std




def AUC(sim_dict, missing_edge_list, non_edge_list):
    if len(missing_edge_list) * len(non_edge_list) <= 10000:
        return auc1(sim_dict, missing_edge_list, non_edge_list)
    else:
        return auc2(sim_dict, missing_edge_list, non_edge_list)
    



def auc1(sim_dict, missing_edge_list, non_edge_list):

    n1 = 0
    n2 = 0
    
    for (u, v) in missing_edge_list:
        try:
            m_s = int(sim_dict[(u, v)] * 1000000)
        except KeyError:
            m_s = 0
        
        for (x, y) in non_edge_list:
            try:
                n_s = int(sim_dict[(x, y)] * 1000000)
            except KeyError:
                n_s = 0
            

            if m_s > n_s:
                n1 += 1
            elif m_s == n_s:
                n2 += 1
            

    n = len(missing_edge_list) * len(non_edge_list)
    return (n1 + 0.5 * n2) / n        


def auc2(sim_dict, missing_edge_list, non_edge_list):

    n = 10000
    n1 = 0
    n2 = 0
    
    m_num = len(missing_edge_list)
    n_num = len(non_edge_list)
    
    for i in range(n):
        r1 = random.randint(0, m_num - 1)
        r2 = random.randint(0, n_num - 1)
        
        (u, v) = missing_edge_list[r1]
        (x, y) = non_edge_list[r2]

        try:
            m_s = int(sim_dict[(u, v)] * 1000000)
        except KeyError:
            m_s = 0
        

        try:
            n_s = int(sim_dict[(x, y)] * 1000000)
        except KeyError:
            n_s = 0
        

        if m_s > n_s:
            n1 += 1
        elif m_s == n_s:
            n2 += 1
        
                
    return (n1 + 0.5 * n2) / n        



def Precision(sim_list, missing_edge_list, missing_edge_num):    

    missing_edge_set = set(missing_edge_list)    

    pre_list = []

    count = 0
    ll = len(sim_list)
    for l in range(missing_edge_num):
        if l < ll:
            (u, v) = sim_list[l][0]
            if (u, v) in missing_edge_set:
                count += 1
            

        if (l + 1) % 10 == 0 and l < 100:  
            pre_list.append(count / (l + 1))
        
    pre_list.append(count / missing_edge_num)

    return pre_list

    

def Ranking_score(sim_list, missing_edge_list, non_edge_num):

    missing_edge_num = len(missing_edge_list)
    
    H = missing_edge_num + non_edge_num
     
    rank_dict = {}
    
    for r in range(len(sim_list)):
        (u, v) = sim_list[r][0]
        rank_dict[(u, v)] = r + 1
    
    
    rr = H - 1
    
    sum_rank = 0
    for (u, v) in missing_edge_list:
        try:
            rank = rank_dict[(u, v)]
        except KeyError:
            rank = rr   
        
        sum_rank += rank
    
            
    return sum_rank / (missing_edge_num * H)


