<!--
 * @Date: 2022-04-30 16:20:28
 * @LastEditors: yuhhong
 * @LastEditTime: 2022-05-07 11:32:16
-->
# I529 - Experiments on DrugCell



<img src="./img/drugcell_graph.png">

This is the final project of I-529. We did the following experiments based on DrugCell:

- **Exp1**: Using unhashed fingerprints of drugs;
- **Exp2**: Using Graph Convolution Networks (GCN/GAT) to embed drugs; To traina GCN/GAT in batchwise: We build up the model parallelly shown in following figure, referring this [issue](https://github.com/tkipf/gcn/issues/4). 
  <img src="./img/barchwised_gcn.png">
- **Eval**: More metrics: mean mean squared error (MSE), pearson correalation (PC). Different features of these two metrics are [here](https://stats.stackexchange.com/questions/314339/should-i-evaluate-my-regression-algorithm-using-mse-or-correlation).

The results are: 

| Mark       | Model                                            | PC        | MSE      | Scripts                                            |
|------------|--------------------------------------------------|-----------|----------|----------------------------------------------------|
| Baseline   | Pretrained model*                                | 0.822805  | 0.014052 | `test_pretrain.sh`                                 |
| Baseline   | Train on `drugcell_all.txt`                      | 0.808271  |          | `ours_train.sh` & `ours_test.sh`                   |
| Exp1       | Train on `drugcell_all.txt` & using unhashed FP  | 0.807748  |          | `ours_train_unhash.sh` & `ours_test_unhash.sh`     |
| Exp2       | Train on `drugcell_all.txt` & GCN                |           |          | coming soon...                                     |
| Exp2       | Train on `drugcell_all.txt` & GAT                |           |          | coming soon...                                     |
| Baseline   | Train on `drugcell_train.txt`                    | 0.315630  | 0.282851 | `commandline_train.sh` & `commandline_test_gpu.sh` |
| Exp2       | Train on `drugcell_train.txt` & GCN              |           |          | coming soon...                                     |
| Exp2       | Train on `drugcell_train.txt` & GAT              | -0.023885 | 0.040629 | `ours_train_graph.sh` & `ours_test_graph.sh`       |

The pretrained model can be downloaded [here](http://drugcell.ucsd.edu/downloads). Our model perfroms better in MSE, but not good in PC. 



## Dataset

The whole dataset can be download [here](http://drugcell.ucsd.edu/downloads).

```bash
$ cat drugcell_all.txt | wc -l
509294
$ cat drugcell_train.txt | wc -l
10000
$ cat drugcell_test.txt | wc -l
1000
```



## Experiments

Please set up the environment as described in `./DrugCell_README.md`. Then install `rdkit` for loading drug graph and `tqdm` for showing the process bar by following command:  

```bash
conda activate pytorch3drugcell
conda install -c rdkit rdkit
conda install -c conda-forge tqdm
```

All the experiments' scripts are in `./sample/`. Please run them as following example: 

```bash
conda activate pytorch3drugcell
cd sample

# test the pretrained model
./test_pretrain.sh

# train and test our own model
./ours_train.sh
./ours_test.sh

# exp1
./ours_train_unhash.sh
./ours_test_unhash.sh

# exp2
./ours_train_graph.sh
./ours_test_graph.sh
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