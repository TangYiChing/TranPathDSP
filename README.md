# TranPathDSP
Molecular Pathways Enhances Drug Response Prediction using Transfer Learning from Cell lines to Tumors and Patient-Derived Xenografts


# How to download TranPathDSP?

```{python}
# a. install TranPathDSP via git clone
$ git clone https://github.com/TangYiChing/TranPathDSP

# b. install TranPathDSP to a conda environment 
$ pip install -r requirements.txt
```

# How to run TranPathDSP?

```{python}

# a. transfer from cell lines to tumors
<> note: please download GDSC.PID_REACTOMEModel.h5 at: https://doi.org/10.5281/zenodo.6093818
$ python ./script/TranPathDSP.py -data ./input_data/Tumor.CHEM-DGNet.EXP.pkl -pretrained ./Tumor.GDSC.PretrainedModel/GDSC.PID_REACTOME.PretrainedModel.h5 -f PID_REACTOME -param ./data/Tumor.bayes_opt.best_params.txt -o Tumor

# b. transfer from cell lines to PDX-D 
# note: please download GDSC.PID.PretrainedModel.h5 at: https://doi.org/10.5281/zenodo.6093818
$ python ./script/TranPathDSP.py -data ./input_data/PDX_D.CHEM-DGNet.EXP.pkl -pretrained ./PDX_D.GDSC.PretrainedModel/GDSC.PID.PretrainedModel.h5 -f PID -param ./data/PDX_D.bayes_opt.best_params.txt -o PDX_D

# c. transfer from cell lines to PDX-C
# note: please download GDSC.PID.PretrainedModel.h5 at: https://doi.org/10.5281/zenodo.6093818
$ python ./script/TranPathDSP.py -data ./input_data/PDX_C.CHEM-DGNet.EXP.pkl -pretrained ./PDX_C.GDSC.PretrainedModel/GDSC.PID.PretrainedModel.h5 -f PID -param ./data/PDX_C.bayes_opt.best_params.txt -o PDX_C
```
