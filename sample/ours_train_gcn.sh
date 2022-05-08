###
 # @Date: 2022-04-28 13:01:14
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-05-07 21:43:52
### 
#!/bin/bash
inputdir="../data/"
gene2idfile=$inputdir"gene2ind.txt"
cell2idfile=$inputdir"cell2ind.txt"
drug2idfile=$inputdir"drug2ind.txt"
traindatafile=$inputdir"drugcell_all_cut.txt"
valdatafile=$inputdir"drugcell_val.txt"
ontfile=$inputdir"drugcell_ont.txt"

mutationfile=$inputdir"cell2mutation.txt"
drugfile=$inputdir"drug2fingerprint.txt"

cudaid=0

modeldir="./exp1_2/Model_sample"
mkdir -p $modeldir

# python -u ../code/train_drugcell.py -onto $ontfile -gene2id $gene2idfile -drug2id $drug2idfile -cell2id $cell2idfile -train $traindatafile -test $valdatafile -model $modeldir -cuda $cudaid -genotype $mutationfile -fingerprint $drugfile -genotype_hiddens 6 -drug_hiddens '100,50,6' -final_hiddens 6 -epoch 100 -batchsize 5000 > train_sample.log
python -u ../code/train_our_drugcell_graph.py -onto $ontfile -drug_graph 'gcn' -gene2id $gene2idfile -drug2id $drug2idfile -cell2id $cell2idfile -train $traindatafile -test $valdatafile -model $modeldir -cuda $cudaid -genotype $mutationfile -fingerprint $drugfile -genotype_hiddens 6 -drug_hiddens '21,21,6' -final_hiddens 50 -batchsize 20 -epoch 100 -lr 0.001
