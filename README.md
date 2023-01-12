# TranPathDSP
Molecular Pathways Enhances Drug Response Prediction using Transfer Learning from Cell lines to Tumors and Patient-Derived Xenografts


# How to download TranPathDSP?

```{python}
# a. install TranPathDSP via git clone
$ git clone https://github.com/TangYiChing/TranPathDSP
```

# How to run TranPathDSP?

```{python}

# a. transfer from cell lines to tumors (change -g 0 to other GPU number that is available on your system)
# note: please download GDSC.PID_REACTOMEModel.h5 at: https://doi.org/10.5281/zenodo.6093818
$ python ./script/TranPathDSP.py -g 0 -data ./input_data/Tumor.CHEM-DGNet-EXP.pkl -pretrained ./Tumor.GDSC.PretrainedModel/GDSC.PID_REACTOME.PreTrainedModel.h5 -f PID_REACTOME -param ./Tumor.bayes_opt.best_params.txt -o Tumor -task classification

# b. transfer from cell lines to PDX-D (change -g 0 to other GPU number that is available on your system)
# note: please download GDSC.PID.PretrainedModel.h5 at: https://doi.org/10.5281/zenodo.6093818
$ python ./script/TranPathDSP.py -g 0 -data ./input_data/PDX_D.CHEM-DGNet.EXP.pkl -pretrained ./PDX_D.GDSC.PretrainedModel/GDSC.PID.PretrainedModel.h5 -f PID -param ./data/PDX_D.bayes_opt.best_params.txt -o PDX_D -task classification

# c. transfer from cell lines to PDX-C (change -g 0 to other GPU number that is available on your system)
# note: please download GDSC.PID.PretrainedModel.h5 at: https://doi.org/10.5281/zenodo.6093818
$ python ./script/TranPathDSP.py -g 0 -data ./input_data/PDX_C.CHEM-DGNet.EXP.pkl -pretrained ./PDX_C.GDSC.PretrainedModel/GDSC.PID.PretrainedModel.h5 -f PID -param ./data/PDX_C.bayes_opt.best_params.txt -o PDX_C -task regression
```
