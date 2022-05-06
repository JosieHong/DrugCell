import sys
import torch
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem


def pearson_corr(x, y):
	xx = x - torch.mean(x)
	yy = y - torch.mean(y)

	return torch.sum(xx*yy) / (torch.norm(xx, 2)*torch.norm(yy,2))

def mean_squard_error(x, y):
	return ((x - y)**2).mean()

def load_ontology(file_name, gene2id_mapping):

	dG = nx.DiGraph()
	term_direct_gene_map = {}
	term_size_map = {}

	file_handle = open(file_name)

	gene_set = set()

	for line in file_handle:

		line = line.rstrip().split()
		
		if line[2] == 'default':
			dG.add_edge(line[0], line[1])
		else:
			if line[1] not in gene2id_mapping:
				continue

			if line[0] not in term_direct_gene_map:
				term_direct_gene_map[ line[0] ] = set()

			term_direct_gene_map[line[0]].add(gene2id_mapping[line[1]])

			gene_set.add(line[1])

	file_handle.close()

	print('There are', len(gene_set), 'genes')

	for term in dG.nodes():
		
		term_gene_set = set()

		if term in term_direct_gene_map:
			term_gene_set = term_direct_gene_map[term]

		deslist = nxadag.descendants(dG, term)

		for child in deslist:
			if child in term_direct_gene_map:
				term_gene_set = term_gene_set | term_direct_gene_map[child]

		# jisoo
		if len(term_gene_set) == 0:
			print('There is empty terms, please delete term:', term)
			sys.exit(1)
		else:
			term_size_map[term] = len(term_gene_set)

	leaves = [n for n in dG.nodes if dG.in_degree(n) == 0]
	#leaves = [n for n,d in dG.in_degree() if d==0]

	uG = dG.to_undirected()
	connected_subG_list = list(nxacc.connected_components(uG))

	print('There are', len(leaves), 'roots:', leaves[0])
	print('There are', len(dG.nodes()), 'terms')
	print('There are', len(connected_subG_list), 'connected componenets')

	if len(leaves) > 1:
		print('There are more than 1 root of ontology. Please use only one root.')
		sys.exit(1)
	if len(connected_subG_list) > 1:
		print( 'There are more than connected components. Please connect them.')
		sys.exit(1)

	return dG, leaves[0], term_size_map, term_direct_gene_map


def load_train_data(file_name, cell2id, drug2id):
	feature = []
	label = []

	with open(file_name, 'r') as fi:
		for line in fi:
			tokens = line.strip().split('\t')

			feature.append([cell2id[tokens[0]], drug2id[tokens[1]]])
			label.append([float(tokens[2])])

	return feature, label


def prepare_predict_data(test_file, cell2id_mapping_file, drug2id_mapping_file):

	# load mapping files
	cell2id_mapping = load_mapping(cell2id_mapping_file)
	drug2id_mapping = load_mapping(drug2id_mapping_file)

	test_feature, test_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping)

	print('Total number of cell lines = %d' % len(cell2id_mapping))
	print('Total number of drugs = %d' % len(drug2id_mapping))

	return (torch.Tensor(test_feature), torch.Tensor(test_label)), cell2id_mapping, drug2id_mapping


def load_mapping(mapping_file):

	mapping = {}

	file_handle = open(mapping_file)

	for line in file_handle:
		line = line.rstrip().split()
		mapping[line[1]] = int(line[0])

	file_handle.close()
	
	return mapping


def prepare_train_data(train_file, test_file, cell2id_mapping_file, drug2id_mapping_file):

	# load mapping files
	cell2id_mapping = load_mapping(cell2id_mapping_file)
	drug2id_mapping = load_mapping(drug2id_mapping_file)

	train_feature, train_label = load_train_data(train_file, cell2id_mapping, drug2id_mapping)
	test_feature, test_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping)

	print('Total number of cell lines = %d' % len(cell2id_mapping))
	print('Total number of drugs = %d' % len(drug2id_mapping))

	return (torch.Tensor(train_feature), torch.FloatTensor(train_label), torch.Tensor(test_feature), torch.FloatTensor(test_label)), cell2id_mapping, drug2id_mapping


def build_input_vector(input_data, cell_features, drug_features): 
	genedim = len(cell_features[0,:])
	drugdim = len(drug_features[0,:])
	feature = np.zeros((input_data.size()[0], (genedim+drugdim)))

	for i in range(input_data.size()[0]):
		feature[i] = np.concatenate((cell_features[int(input_data[i,0])], drug_features[int(input_data[i,1])]), axis=None)

	feature = torch.from_numpy(feature).float()
	return feature


# -----------------------------------------
# Our own utils
# -----------------------------------------

def load_our_drug_fp(drug2id_mapping_file): 
	
	with open(drug2id_mapping_file, 'r') as f:
		smiles_list = [d.split()[1] for d in f.read().split('\n') if d != '']

	drug_features = []
	for smiles in smiles_list:
		mol = Chem.MolFromSmiles(smiles)
		fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
		# fp = AllChem.GetHashedMorganFingerprint(mol, 2)
		drug_features.append(np.array(fp))

	return np.array(drug_features)

def load_our_drug_graph_features(drug2id_mapping_file, max_atom_num=300): 
	
	with open(drug2id_mapping_file, 'r') as f:
		smiles_list = [d.split()[1] for d in f.read().split('\n') if d != '']

	drug_graphs = []
	drug_features = []
	
	for smiles in smiles_list:
		mol = Chem.MolFromSmiles(smiles)
		drug_graphs.append(create_graph(mol, max_atom_num))
		drug_features.append(create_feature(mol, max_atom_num))

	return np.array(drug_graphs), np.array(drug_features)

def create_graph(mol, size=50): 
	A = np.zeros((size, size))
	for atom in mol.GetAtoms():
		i = atom.GetIdx()
		for neighbor in atom.GetNeighbors():
			j = neighbor.GetIdx()
			A[i, j] = 1
			A[j, i] = 1
	return A

def create_feature(mol, size=50): 
	atom_map = {'C': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'H': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
				'O': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'N': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
				'F': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'S': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
				'Cl': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'P': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
				'B': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'Br': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
				'I': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'Pt': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
				'As': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
	X = np.zeros((size, 21))
	for atom in mol.GetAtoms():
		X[atom.GetIdx(), 0] = atom.GetMass() # 1. atomic mass;
		X[atom.GetIdx(), 1] = atom.GetTotalNumHs() # 2. total number of Hs (explicit and implicit);
		X[atom.GetIdx(), 2] = atom.GetIsAromatic() # 3. Is aromatic;
		X[atom.GetIdx(), 3] = atom.GetExplicitValence() # 4. valence minus the number of hydrogens;
		X[atom.GetIdx(), 4] = atom.GetDegree() # 5. number of immediate neighbors who are “heavy” (nonhydrogen) atoms;
		X[atom.GetIdx(), 5] = atom.GetFormalCharge() # 6. atomic charge;
		X[atom.GetIdx(), 6] = atom.GetNumImplicitHs() # 7. number of implicit hydrogens;
		X[atom.GetIdx(), 7] = int(atom.IsInRing()) # 8. Is in a ring 
		X[atom.GetIdx(), 8:] = np.array(atom_map[atom.GetSymbol()])
	return X

def build_input_seperately_batched(input_data, cell_features, graph_features, drug_features, batch_size): 
	'''
	input_data size: [batch size, 2]
	'''
	batch_size, _ = input_data.shape

	_, celldim = cell_features.shape
	cellf = np.zeros((batch_size, celldim))

	_, _, graphdim = graph_features.shape
	graphf = np.zeros((graphdim*batch_size, graphdim*batch_size))

	_, atomdim, drugdim = drug_features.shape
	drugf = np.zeros((atomdim*batch_size, drugdim*batch_size))
	# print(cell_features.shape, graph_features.shape, drug_features.shape)
	# (1225, 3008) (684, 300, 300) (684, 300, 4)
	# print(cellf.shape, graphf.shape, drugf.shape)
	# (4, 3008) (1200, 1200) (1200, 32)

	for i in range(input_data.size()[0]): 
		cellf[i] = np.array(cell_features[int(input_data[i,0])])
		graphf[i*graphdim:(i+1)*graphdim, i*graphdim:(i+1)*graphdim] = np.array(graph_features[int(input_data[i,1])])
		drugf[i*atomdim:(i+1)*atomdim, i*drugdim:(i+1)*drugdim] = np.array(drug_features[int(input_data[i,1])])

	cellf = torch.from_numpy(cellf).float()
	graphf = torch.from_numpy(graphf).float()
	drugf = torch.from_numpy(drugf).float()
	# print(cellf.size(), graphf.size(), drugf.size())
	# torch.Size([4, 3008]) torch.Size([1200, 1200]) torch.Size([1200, 32])
	return cellf, graphf, drugf


def build_input_seperately(input_data, cell_features, graph_features, drug_features): 
	'''
	It should be noticed that because of the GCN and GAT, we will 
	not load the batched data here. 
	'''
	cellf = np.array(cell_features[int(input_data[0])])
	graphf= np.array(graph_features[int(input_data[1])])
	drugf = np.array(drug_features[int(input_data[1])])

	cellf = torch.from_numpy(cellf).float()
	graphf = torch.from_numpy(graphf).float()
	drugf = torch.from_numpy(drugf).float()
	return cellf, graphf, drugf
