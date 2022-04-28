# optimize model hyperparameters via Bayesian Optimization: 

step1. install bayesopt: https://github.com/rmcantin/bayesopt

step2. download pretrained cell line model at: https://doi.org/10.5281/zenodo.6093818

step3. execute the following python script

```{python}

# tune parameters
$ python ./script/BayesOpt.py -train CHEM-DGNet-EXP.CV_Fold0.train.pkl -valid CHEM-DGNet-EXP.CV_Fold0.valid.pkl -test CHEM-DGNet-EXP.CV_Fold0.test.pkl -f PID -pretrained ./Tumor.GDSC.PretrainedModel/GDSC.PID.PretrainedModel.h5 -task classification -o Tumor.PID  
```
