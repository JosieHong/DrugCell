import numpy as np
import torch
import torch.utils.data as du
import torch.nn as nn
from util import *
from drugcell_Graph import drugcell_graph
import argparse
import numpy as np
import time
from tqdm import tqdm

# build mask: matrix (nrows = number of relevant gene set, ncols = number all genes)
# elements of matrix are 1 if the corresponding gene is one of the relevant genes
def create_term_mask(term_direct_gene_map, gene_dim):

	term_mask_map = {}

	for term, gene_set in term_direct_gene_map.items():

		mask = torch.zeros(len(gene_set), gene_dim)

		for i, gene_id in enumerate(gene_set):
			mask[i, gene_id] = 1

		mask_gpu = torch.autograd.Variable(mask.cuda(CUDA_ID))

		term_mask_map[term] = mask_gpu

	return term_mask_map

 
def train_model(root, term_size_map, term_direct_gene_map, drug_graph, dG, train_data, gene_dim, drug_dim, atom_num, model_save_folder, train_epochs, batch_size, learning_rate, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, cell_features, drug_graphs, drug_features):

	epoch_start_time = time.time()
	best_model = 0
	max_corr = 0

	# dcell neural network
	model = drugcell_graph(term_size_map, term_direct_gene_map, dG, drug_graph, gene_dim, drug_dim, atom_num, root, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, batch_size, device=CUDA_ID)

	train_feature, train_label, test_feature, test_label = train_data

	train_label_gpu = torch.autograd.Variable(train_label.cuda(CUDA_ID))
	test_label_gpu = torch.autograd.Variable(test_label.cuda(CUDA_ID))

	model.cuda(CUDA_ID)

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
	term_mask_map = create_term_mask(model.term_direct_gene_map, gene_dim)

	optimizer.zero_grad()

	for name, param in model.named_parameters():
		term_name = name.split('_')[0]

		if '_direct_gene_layer.weight' in name:
			param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
		else:
			param.data = param.data * 0.1
	
	# non-batch
	# train_loader = du.DataLoader(du.TensorDataset(train_feature,train_label), batch_size=None, shuffle=True)
	# test_loader = du.DataLoader(du.TensorDataset(test_feature,test_label), batch_size=None, shuffle=False)
	# batch
	# j0sie: In theory, we need to `drop_last=True`, because we will concatenate the graph features. 
	# But, when `drop_last=True`, `pearson_corr` is weird. So we chooes a proper batch size, and set 
	# `drop_last=False`.  
	assert train_feature.shape[0] % batch_size == 0 
	assert test_feature.shape[0] % batch_size == 0 
	train_loader = du.DataLoader(du.TensorDataset(train_feature,train_label), batch_size=batch_size, shuffle=False) 
	test_loader = du.DataLoader(du.TensorDataset(test_feature,test_label), batch_size=batch_size, shuffle=False) 

	for epoch in range(train_epochs):

		#Train
		model.train()
		train_predict = torch.zeros(0,0).cuda(CUDA_ID)
		# train_label = torch.zeros(0,0).cuda(CUDA_ID)

		with tqdm(total=len(train_loader)) as bar:
			for i, (inputdata, labels) in enumerate(train_loader):
				# Convert torch tensor to Variable
				# non-batch
				# cellf, graphf, drugf = build_input_seperately(inputdata, cell_features, drug_graphs, drug_features) # inputdata are drugid and cellid
				# batch
				cellf, graphf, drugf = build_input_seperately_batched(inputdata, cell_features, drug_graphs, drug_features, batch_size) # inputdata are drugid and cellid

				cuda_cellf = torch.autograd.Variable(cellf.cuda(CUDA_ID))
				cuda_graphf = torch.autograd.Variable(graphf.cuda(CUDA_ID))
				cuda_drugf = torch.autograd.Variable(drugf.cuda(CUDA_ID))
				cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))
				# non-batch
				# Because of GCN/GAT in drug embedding, we can not load the batched data. 
				# Let's recover `gene_input` to `batch_size=1`.
				# cuda_labels = cuda_labels.unsqueeze(0)

				# Forward + Backward + Optimize
				optimizer.zero_grad()  # zero the gradient buffer

				# Here term_NN_out_map is a dictionary 
				aux_out_map, _ = model(cuda_cellf, cuda_drugf, cuda_graphf)

				if train_predict.size()[0] == 0:
					train_predict = aux_out_map['final'].data
					# train_label = cuda_labels
				else:
					train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)
					# train_label = torch.cat([train_label, cuda_labels], dim=0)

				# j0sie: why calculate loss on every layer? 
				total_loss = 0	
				for name, output in aux_out_map.items():
					loss = nn.MSELoss()
					if name == 'final': 
						total_loss += loss(output, cuda_labels)
					else: # change 0.2 to smaller one for big terms
						total_loss += 0.2 * loss(output, cuda_labels)
				# loss = nn.MSELoss()
				# total_loss = loss(aux_out_map['final'], cuda_labels)

				# update bar
				bar.set_description('Train')
				bar.set_postfix(loss=total_loss.item())
				bar.update(1)

				total_loss.backward()

				for name, param in model.named_parameters():
					if '_direct_gene_layer.weight' not in name:
						continue
					term_name = name.split('_')[0]
					#print name, param.grad.data.size(), term_mask_map[term_name].size()
					param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

				optimizer.step()

		# train_corr = pearson_corr(train_predict, train_label)
		# train_mse = mean_squard_error(train_predict, train_label)
		train_corr = pearson_corr(train_predict, train_label_gpu)
		train_mse = mean_squard_error(train_predict, train_label_gpu)

		#if epoch % 10 == 0:
		torch.save(model, model_save_folder + '/model_' + str(epoch) + '.pt')

		#Test: random variables in training mode become static
		model.eval()
		
		test_predict = torch.zeros(0,0).cuda(CUDA_ID)
		# test_label = torch.zeros(0,0).cuda(CUDA_ID)

		with tqdm(total=len(test_loader)) as bar:
			for i, (inputdata, labels) in enumerate(test_loader):
				# Convert torch tensor to Variable
				# non-batch
				# cellf, graphf, drugf = build_input_seperately(inputdata, cell_features, drug_graphs, drug_features) # inputdata are drugid and cellid
				# batch
				cellf, graphf, drugf = build_input_seperately_batched(inputdata, cell_features, drug_graphs, drug_features, batch_size) # inputdata are drugid and cellid

				cuda_cellf = torch.autograd.Variable(cellf.cuda(CUDA_ID), requires_grad=False)
				cuda_graphf = torch.autograd.Variable(graphf.cuda(CUDA_ID), requires_grad=False)
				cuda_drugf = torch.autograd.Variable(drugf.cuda(CUDA_ID), requires_grad=False)
				cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID), requires_grad=False)

				aux_out_map, _ = model(cuda_cellf, cuda_drugf, cuda_graphf)

				bar.set_description('Eval')
				bar.update(1)

				if test_predict.size()[0] == 0:
					test_predict = aux_out_map['final'].data
					# test_label = cuda_labels
				else: 
					test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)
					# test_label = torch.cat([test_label, cuda_labels], dim=0)

		# test_corr = pearson_corr(test_predict, test_label)
		# test_mse = mean_squard_error(test_predict, test_label)
		test_corr = pearson_corr(test_predict, test_label_gpu)
		test_mse = mean_squard_error(test_predict, test_label_gpu)

		epoch_end_time = time.time()
		print("epoch\t%d\tcuda_id\t%d\ttrain_corr\t%.6f\ttrain_mse\t%.6f\tval_corr\t%.6f\tval_mse\t%.6f\ttotal_loss\t%.6f\telapsed_time\t%s" % (epoch, CUDA_ID, train_corr, train_mse, test_corr, test_mse, total_loss, epoch_end_time-epoch_start_time))
		epoch_start_time = epoch_end_time
	
		if test_corr >= max_corr:
			max_corr = test_corr
			best_model = epoch

	torch.save(model, model_save_folder + '/model_final.pt')	

	print("Best performed model (epoch)\t%d" % best_model)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train dcell')
	parser.add_argument('-onto', help='Ontology file used to guide the neural network', type=str)
	parser.add_argument('-drug_graph', help='Graph model embedding drugs', type=str, choices=['gcn', 'gat'])
	parser.add_argument('-train', help='Training dataset', type=str)
	parser.add_argument('-test', help='Validation dataset', type=str)
	parser.add_argument('-epoch', help='Training epochs for training', type=int, default=300)
	parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)
	parser.add_argument('-batchsize', help='Batchsize', type=int, default=5000)
	parser.add_argument('-modeldir', help='Folder for trained models', type=str, default='MODEL/')
	parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)
	parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str)
	parser.add_argument('-drug2id', help='Drug to ID mapping file', type=str)
	parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str)
	parser.add_argument('-atomnum', help='number of atoms in each drug', type=int, default=300)

	parser.add_argument('-genotype_hiddens', help='Mapping for the number of neurons in each term in genotype parts', type=int, default=6)
	parser.add_argument('-drug_hiddens', help='Mapping for the number of neurons in each layer', type=str, default='100,50,6')
	parser.add_argument('-final_hiddens', help='The number of neurons in the top layer', type=int, default=6)

	parser.add_argument('-genotype', help='Mutation information for cell lines', type=str)
	parser.add_argument('-fingerprint', help='Morgan fingerprint representation for drugs', type=str)

	# call functions
	opt = parser.parse_args()
	torch.set_printoptions(precision=5)

	# load input data
	train_data, cell2id_mapping, drug2id_mapping = prepare_train_data(opt.train, opt.test, opt.cell2id, opt.drug2id)
	gene2id_mapping = load_mapping(opt.gene2id)

	# load cell/drug features
	cell_features = np.genfromtxt(opt.genotype, delimiter=',')
	drug_graphs, drug_features = load_our_drug_graph_features(opt.drug2id, opt.atomnum)
	# print(drug_graphs.shape, drug_features.shape) # (684, 300, 300) (684, 300, 4)

	num_cells = len(cell2id_mapping)
	num_drugs = len(drug2id_mapping)
	num_genes = len(gene2id_mapping)
	drug_dim = len(drug_features[0,0,:])

	# load ontology
	dG, root, term_size_map, term_direct_gene_map = load_ontology(opt.onto, gene2id_mapping)

	# load the number of hiddens #######
	num_hiddens_genotype = opt.genotype_hiddens

	num_hiddens_drug = list(map(int, opt.drug_hiddens.split(',')))

	num_hiddens_final = opt.final_hiddens
	#####################################

	CUDA_ID = opt.cuda

	train_model(root, term_size_map, term_direct_gene_map, opt.drug_graph, dG, train_data, num_genes, drug_dim, opt.atomnum, opt.modeldir, opt.epoch, opt.batchsize, opt.lr, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, cell_features, drug_graphs, drug_features)

