# NAGC-project
In this thesis, the topics that were analyzed were the problem of community detection
in graphs with node attributes and the way the New Attributed Graph Clustering (NAGC)
method can provide a solution to this problem. Experiments on preparation and appropriate
transformation of data were performed and evaluated with known metrics and the mathematical
background of the optimization methods used was analyzed, as well. The aim of the assignment
is to optimize the performance of the NAGC method, regarding the accuracy of the communities
it detects and also, to investigate alternative ways of data management and how this can affect
the final result. Further on this thesis, an analysis is conducted, regarding the initialization and
normalization methods performed on the data, which are introduced in the form of matrices,
the factorization and completion techniques of these matrices, the optimization process used,
the type and structure of the data samples, as well as the results of the final algorithm, which
are examined from different points of view.The initial code has been edited, in order to serve the
purposes of this work.

Datasets

1. WebKB: Consists of 877 websites (nodes) from four universities related to computer science : Cornell university, Texas university, Washington university and Wisconsin university.The websites are connected with 1480 hyperlinks (edges). The hyperlinks are classified to: Course, Faculty, Student, Project, Staff. It contains 1703 unique words.
2. Cora: Consists of 2708 scientific papers (nodes) related to machine learning, which are connected with 5278 citation links (edges). The edges are classified to: Case Based, Genetic Algorithms, Neural Networks, Probabilistic Methods, Reinforcement Learning, Rule Learning, Theory. It contains 1433 unique words.
3. Citeseer: Consist of 3312 papers, which are connected with 4660 citation links (edges). The papers are classified to: Agents, AI, DB, IR, ML, HCI. It contains 3703 unique words.
4. PolBlog: Consists of 1490 websites related to political content of the USA, which are connected with 16630 hyperlinks (edges). The websites are classified to: Liberal , Conservative. It contains 7 unique words.


.py files

• build_graph : Takes as input the graph data in .cite or .content format and creates matrices S (adjacency matrix) and X (node-attribute matrix). Also, it saves the groundtruth node to cluster membership.
• init_kmeans : Defines the k-means ++ method.
• init_agglomerative : Defines the agglomerative method.
• init_spectral : Defines the spectral method.
• NAGC_class : Optimizes the algorithm with the defined updating rules (&PU learning - matrix completion). Also, in this part we include the definition of the transition sigmoid function.
• NAGC_ReLU : Optimizes the algorithm with the defined updating rules (&PU learning - matrix completion). Also, in this part we include the definition of the transition RelU function.
• VU_init : Initialises the matrices U, and V with regard to the selected initialization method as it is defiined in init_kmeans, init_agglomerative, init_spectral and init_dbscan.
• evaluate :Defines the evaluation metrics ARI, entropy, modularity.
• main : Executes the algorithm and provides the metric results.


