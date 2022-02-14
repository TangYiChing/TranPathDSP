# TransPathDSP
Molecular Pathways Enhances Drug Response Prediction using Transfer Learning from Cell lines to Tumors and Patient-Derived Xenografts


# How to download TranPathDSP?

```{python}
# a. install TranPathDSP via git clone
$ git clone https://github.com/TangYiChing/TranPathDSP

# b. install TranPathDSP to a conda environment 
$ pip install -r requirements.txt
```

# How to run TranPathDSP?

## Note. before running the following code, please download the corresponding pretrained cell line models at:

```{python}

# a. transfer from cell lines to tumors
$ python ./script/TranPathDSP_CV.py -data ./input_data/Tumor.CHEM-DGNet.EXP.pkl -pretrained ./gdsc_celllinemodel/Tumor.Pretrained.CellLineModel.h5 -f PID_REACTOME -param ./data/Tumor.bayes_opt.best_params.txt -o Tumor

# b. transfer from cell lines to PDX-D 
$ python ./script/TranPathDSP_CV.py -data ./input_data/PDX_D.CHEM-DGNet.EXP.pkl -pretrained ./gdsc_celllinemodel/PDX_D.Pretrained.CellLineModel.h5 -f PID -param ./data/PDX_D.bayes_opt.best_params.txt -o PDX_D

# c. transfer from cell lines to PDX-C
$ python ./script/TranPathDSP_CV.py -data ./input_data/PDX_C.CHEM-DGNet.EXP.pkl -pretrained ./gdsc_celllinemodel/PDX_C.Pretrained.CellLineModel.h5 -f PID -param ./data/PDX_C.bayes_opt.best_params.txt -o PDX_C
```
