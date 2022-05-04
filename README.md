<!--
 * @Date: 2022-04-30 16:20:28
 * @LastEditors: yuhhong
 * @LastEditTime: 2022-05-03 22:45:12
-->
# I529 - Experiments on DrugCell



<img src="./img/drugcell_graph.png" width="800">

This is the final project of I-529. We did the following experiments based on DrugCell:

- Using unhashed fingerprints of drugs;
- Using Graph Convolution Network (GCN) to embed drugs.  

The results are: 

| Model                                            | Test Pearson Corr |
|--------------------------------------------------|-------------------|
| Pretrained model                                 | 0.822805          |
| Train on `drugcell_all.txt`                      | 0.808271          |
| Train on `drugcell_all.txt` & using unhashed FP  | 0.807748          |
| Train on `drugcell_train.txt`                    | 0.445488          |
| Train on `drugcell_train.txt` & drug graph (GCN) | 0.234693          |



<!-- ## Setup TorchDrug

```bash
# pytorch
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
# pytorch-scatter
conda install pytorch-scatter -c pyg
# torchdrug
pip install torchdrug
``` -->

<!-- ## Pre 

```
1. Model description (30 pts)
  - DrugCell
    - Data encoding:
      - genotypes are encoded to binary mutations
      - drugs are encoded to a hashed binary vector of Morgan fingerprint
    - Model
  - *graph*:
    - Data encoding:
      - genotypes are encoded to binary mutations
      - drugs are encoded to adjacency matrix and features matrix
    - Using a graph model embedding the drugs

2. Data and Metric (20 pts)
  - Data: the Cancer Therapeutics Response Portal (CTRP) v2 and the Genomics of Drug Sensitivity in Cancer (GDSC) database from DrugCell
  - Metric: Pearson correlation
  
3. Results and Conclusion (40 pts)
  - hash or unhash do not have an obvious effect on prediction
  - expect graph model can embed better than fingerprint

4. Q&A (10 pts)
``` -->


## Dataset

```bash
$ cat drugcell_all.txt | wc -l
509294
$ cat drugcell_train.txt | wc -l
10000
$ cat drugcell_test.txt | wc -l
1000
```



## Train on the whole dataset

The pretrained model and whole dataset can be download [here](http://drugcell.ucsd.edu/downloads).

```bash
# 1. test the pretrained model
./commandline_test_gpu.sh

Total number of cell lines = 1225
Total number of drugs = 684
Total number of genes = 3008
Test pearson corr       GO:0008150      0.822805

# 2. train our own model
./ours_train.sh

# other exp
./ours_train_unhash.sh
./ours_train_graph.sh

# 3. test our own model
./ours_test.sh
```



## Reference

```bib
@article{kuenzi2020predicting,
  title={Predicting drug response and synergy using a deep learning model of human cancer cells},
  author={Kuenzi, Brent M and Park, Jisoo and Fong, Samson H and Sanchez, Kyle S and Lee, John and Kreisberg, Jason F and Ma, Jianzhu and Ideker, Trey},
  journal={Cancer cell},
  volume={38},
  number={5},
  pages={672--684},
  year={2020},
  publisher={Elsevier}
}
```