# optimize model hyperparameters via Bayesian Optimization: 

step1. install bayesopt: https://github.com/rmcantin/bayesopt

step2. download pretrained cell line model at:

step3. execute the following python script

```{python}

# tune parameters
$ python ./script/BayesOpt.py -train -valid -test -f PID -pretrained ./Tumor.GDSC.PretrainedModel/GDSC.PID.PretrainedModel.h5 -task classification -o Tumor.PID  
```
