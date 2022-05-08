###
 # @Date: 2022-04-28 12:56:40
 # @LastEditors: yuhhong
 # @LastEditTime: 2022-05-07 21:44:33
### 
#!/bin/bash
inputdir="../data/"
gene2idfile=$inputdir"gene2ind.txt"
cell2idfile=$inputdir"cell2ind.txt"
drug2idfile=$inputdir"drug2ind.txt"
testdatafile=$inputdir"drugcell_test.txt"

mutationfile=$inputdir"cell2mutation.txt"
drugfile=$inputdir"drug2fingerprint.txt"

modelfile="./exp1_2/Model_sample/model_final.pt"

resultdir="./exp1_2/Result_sample"
hiddendir="./exp1_2/Hidden_sample"

cudaid=$1

if [$cudaid = ""]; then
	cudaid=0
fi

mkdir -p $resultdir
mkdir -p $hiddendir

python -u ../code/predict_our_drugcell_graph.py -gene2id $gene2idfile -cell2id $cell2idfile -drug2id $drug2idfile -genotype $mutationfile -fingerprint $drugfile -hidden $hiddendir -result $resultdir -predict $inputdir/drugcell_test.txt -load $modelfile -cuda $cudaid -batchsize 10