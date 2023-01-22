# This program build adjacency matrix and attribute matrix from graph data.
# This program can deal with two format(.mat and .cite&.content)
# Outputs are below
'''
S : adjacency matrix with prepocess
S_ori : original adjacency matrix
A : attribute matrix with preprocess
clus : true cluster of nodes
flag : with ground truth or without
A_ori : original attribute matrix
'''

import numpy as np
import glob
import scipy.io

def build_graph(path): # switch for two form of file
    if "mat" in path:
        print ("mat")
        S,S_ori,A,clus,A_ori = for_mat(path)
        flag=1
        if clus==[]:
            flag=0
        return S,S_ori,A,clus,flag,A_ori
    else:
        print ("cite")
        S,S_ori,A,clus,A_ori = for_cites_contents(path)
        return S,S_ori,A,clus,1,A_ori


def for_mat(path):
    mat_contents = scipy.io.loadmat(path)
    G = mat_contents["Network"]
    X = mat_contents["Attributes"]
    Label = list(map(int,mat_contents["Label"]))
    node_size = G.shape[0]
    att_size = X.shape[1]
    # S = lil_matrix((node_size,node_size))
    # S = np.zeros((node_size,node_size))
    # A = np.zeros((node_size,att_size))
    S = np.zeros((node_size,node_size))
    A = X.toarray()
    # fill the adjacency matrix and attribute matrix
    nonzeros = G.nonzero()
    print ("no.nodes: " + str(node_size))
    print ("no.attributes: " + str(att_size))
    edgecount=0
    for i in range(len(nonzeros[0])):
        S[nonzeros[0][i],nonzeros[1][i]] = 1
        S[nonzeros[1][i],nonzeros[0][i]] = 1
    # erase diagonal elements
    diag = 0
    for i in range(node_size):
        diag += S[i,i]
    nonzeros = S.nonzero()
    edge_count = int((len(nonzeros[0])+diag)/2)
    print ("number of edges : " + str(edge_count))

    S_pre,A_pre=preprocess(S,A)
    return S_pre,S,A_pre,Label,A #scaling S and A


def for_cites_contents(path):
    node={}# key value pairs e.g vaso: 1
    counter=0
    att_list=[]
    clus=[]
############ download atttributes #################
    infiles = glob.glob(path+'/*.content')
    for infile in infiles:
        with open(infile) as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                tmp = line.split("\t")
                # assign an index to each node
                # in the nodes dictionary go to the
                # node entry and assign the number of the counter (index)
                node[tmp[0]]=counter
                counter+=1
                # get the remaining elements of each line and append to the
                # attributes list (which becomes a list of lists 2-d array)
                att_list.append((tmp[1:-1]))
                # get the last element of each line and add to the end
                # of the list (array like)
                clus.append(tmp[-1].replace("\n",""))
    print ("number of nodes : " + str(len(node)))
    # print the number of attributes of the first entry
    print ("number of attributes : " + str(len(att_list[0])))

############ download edges #################
    edges=[]
    infiles = glob.glob(path+'/*.cites')
    for infile in infiles:
        with open(infile) as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                line = line.replace("\n","")
                tmp = line.split() # when without passing parameters python assumes split on space
                # if each url exists in nodes
                if tmp[0] in node and tmp[1] in node:
                    # get the index of the beginning and end of the edge
                    ind0 = node[tmp[0]]
                    ind1 = node[tmp[1]]
                    # append the edge as a tuple in the edges
                    edges.append((ind0,ind1))
    node_size = len(node) # the number of nodes
    att_size = len(att_list[0]) # the number of attributes

    # create the proximity matrix and initialize all distances to 0
    S = np.zeros((node_size,node_size))

    # create the node attribute matrix and initialize all to 0
    A = np.zeros((node_size,att_size))
    # iterate over the edges and go to the proximity matrix
    # and set distance from node 1 to node 2 to 1
    # and from node 2 to node 1 to 1 (convert to undirected)
    for i in range(len(edges)):
        S[edges[i][0],edges[i][1]] = 1
        S[edges[i][1],edges[i][0]] = 1

    diag = 0
    # for each row of the proximity matrix
    for i in range(node_size):
        # count the self edges (diagonal elements)
        diag += S[i,i]
    # get all edges (both directions and self-edges)
    nonzeros = S.nonzero()
    # Calculate number of edges
    # adding the number of node indexes (rows) that are non-zero (starts of edges)
    # to the number of diagonal elements and divides by two because matrix is symmetric
    # so does not double count edges
    edge_count = int((len(nonzeros[0]) + diag)/2)
    # prints the number
    print ("number of edges : " + str(edge_count))

    # iterates over the attributes of each node
    # len(att_list) gives the number of rows and len(att_list[0]) gives the
    # number of columns, rows are nodes and columns are attributes
    for i in range(len(att_list)):
        for j in range(len(att_list[0])):
            # fills the Attribute matrix (A)
            # with the attributes of each node converted to
            # floating point
            A[i,j] = float(att_list[i][j])
    # A=A.tocsr()
    # print("Maximum value of A:" + str(A.max(axis=1)))
    # print("Maximum value of A:" + str(A.max(axis=0)))
    # print("Minimum value of A:" + str(A.min(axis=1)))
    # print("Minimum value of A:" + str(A.min(axis=0)))

    S_pre,A_pre=preprocess(S,A)

    return S_pre, S, A_pre, clus, A #scaling S and A

def preprocess(S,A):
    # initialization in the paper(JWNMF)
    # S = S / S.sum()
    # A = A / A.sum()

    # Normalization based on size of S
    # A = A * S.sum() / A.sum()

    # Sums over all the elements of A and S
    # and normalized matrix using the following formula
    S = S * A.sum() / S.sum()
    return S, A
