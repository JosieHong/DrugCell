<!--
 * @Date: 2022-04-30 16:20:28
 * @LastEditors: yuhhong
 * @LastEditTime: 2022-05-03 13:44:37
-->
# I529 - DrugCell

This is the final project of I-529. We attempt to do some modification of DrugCell. 

| Model                                           | Test Pearson Corr |
|-------------------------------------------------|-------------------|
| Pretrained model                                | 0.822805          |
| Train on `drugcell_all.txt` & using hashed FP   | 0.808271          |
| Train on `drugcell_all.txt` & using unhashed FP | -                 |
| Train on `drugcell_all.txt` & molecules graph   | -                 |


<!-- ## Setup TorchDrug

```bash
# pytorch
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
# pytorch-scatter
conda install pytorch-scatter -c pyg
# torchdrug
pip install torchdrug
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