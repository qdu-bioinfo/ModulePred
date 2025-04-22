import pandas as pd
import itertools
from tqdm import tqdm 



data1 = pd.read_csv('../data/all_human_complexes.csv')
data_ppi = pd.read_csv('../data/gene_gene.csv')
data_gd = pd.read_csv('../data/disease_gene_edge.csv')

all_g1 = data_ppi['gene1'].tolist() + data_ppi['gene2'].tolist()
all_g2 = data_gd['gene'].tolist()
all_gene = list(set(all_g1 + all_g2))



str_gene = []
for i in tqdm(range(3637)):
    data = data1['subunits(Gene name)'][i]

    values = data.split(';')

    str_gene = str_gene + values

str_p = []
for i in tqdm(range(3637)):
    data = data1['subunits(UniProt IDs)'][i]

    values = data.split(';')

    str_p = str_p + values
    
    

str_gene = []

data_new = data1.copy()

sum = 0
for i in range(3637):
    data = data1['subunits(Gene name)'][i]

    values = data.split(';')

    str_gene = list(set(values) & set(all_g1))
    
    edges = list(itertools.combinations(values, 2))
    
    if len(edges) == 0:
        
        data_new.drop(i, inplace = True)
        sum = sum + 1
        

print('==',sum)


data_new.reset_index(drop=True, inplace=True)



d = data_gd['disease'].tolist()
g = data_gd['gene'].tolist()
edges = list(zip(d, g))

graph = {}
for d, A in edges:
    if A in graph:
        graph[A].append(d)
    else:
        graph[A] = [d]
        
        
        
_list = []
for i in range(3548):
    
    my_tuple = () 
    my_list = []
    list_sum = [] 
    
    data = data_new['subunits(Gene name)'][i]

    values = data.split(';')
    
    for j in values:
        
        if j not in graph.keys():
            continue;
        
        elements = graph[j]
        
        if len(elements) == 0:
            continue;
            
        list_sum = list_sum + elements
        
        
        
    for element in list_sum:
        
        if len(list_sum) == 0:
            
            _list.append(('No'))
            break;
        else:
            my_tuple += (element,)
        
    my_list.append(my_tuple)
    
    new_elements = [",".join(sub_tuple).replace(",", ";") for sub_tuple in my_list]  

    my_tuple = (new_elements[0],)      
            
    _list.append(my_tuple)
    
data_a = pd.DataFrame(_list)
data_a.columns = ['Disease']


data_new['Disease'] = data_a



data_new.columns = ['ComplexID','Organism','subunits','subunits(Gene name)','Disease']




data_new.to_csv('Output/used_human_complexes.csv', index = False)

