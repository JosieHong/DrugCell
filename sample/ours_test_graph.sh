###
 # @Date: 2022-04-28 12:56:40
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-05-05 11:38:05
### 
#!/bin/bash
inputdir="../data/"
gene2idfile=$inputdir"gene2ind.txt"
cell2idfile=$inputdir"cell2ind.txt"
drug2idfile=$inputdir"drug2ind.txt"
testdatafile=$inputdir"drugcell_test.txt"

mutationfile=$inputdir"cell2mutation.txt"
drugfile=$inputdir"drug2fingerprint.txt"

modelfile="./ours_train_graph/Model_sample/model_final.pt"

resultdir="./ours_train_graph/Result_sample"
hiddendir="./ours_train_graph/Hidden_sample"

cudaid=$1

if [$cudaid = ""]; then
	cudaid=0
fi

mkdir -p $resultdir
mkdir -p $hiddendir

python -u ../code/predict_drugcell.py -gene2id $gene2idfile -cell2id $cell2idfile -drug2id $drug2idfile -genotype $mutationfile -fingerprint $drugfile -hidden $hiddendir -result $resultdir -predict $inputdir/drugcell_test.txt -load $modelfile -cuda $cudaid > test_sample.log