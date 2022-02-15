# processed drug structure -- CHEM

```{python}

# given SMILE string, generate molecular fingerprint with size of 256 bits
# note: RDKit is required
# note: please download DB.Drug.Smile.txt at: 
$ python ./script/smile2bit.py DB.Drug.Smile.txt -o Drug
```

# processed drug-target associated network -- DGNet

```{python}

# given gene targets, generate pathway enrichment score 
# note: please install the NetPEA tools at: https://github.com/TangYiChing/NetPEA
# note: please download DB.Drug.Target.txt at:
# note: please download 9606.protein_name.links.v11.0.pkl at:
# note: please download union.c2.cp.pid.reactome.v7.2.symbols.gmt at:
$ python ./NetPEA/NetPEA/run_netpea.py -r DB.Drug.Target.txt -ppi ./STRING/9606.protein_name.links.v11.0.pkl -p ./MSigDB/union.c2.cp.pid.reactome.v7.2.symbols.gmt -o Drug
```

# processed gene expression -- EXP

```{python}
# 1. given gene expression, generate harmonized expression values
# note: please install pycombat at: https://epigenelabs.github.io/pyComBat/
# note: please download *.expression.txt and *.resp.Alias.txt at:
$ python ./run_combat.py -exp GDSC.expression.txt Lee2021.expression.txt GeoSearch.expression.txt Ding2016.expression.txt -anno GDSC.resp.1-AUC.alias.txt Lee2021.resp.Alias.txt GeoSearch.resp.Alias.txt Ding2016.resp.Alias.txt -o Tumor

# 2. given harmonized expression, generate pathway enrichment score
# note: please install ssGSEA at: https://gseapy.readthedocs.io/en/latest/introduction.html
# note: please downlowd union.c2.cp.pid.reactome.v7.2.symbols.gmt at: 
$ python ./run_ssGSEA.py -exp Tumor.combat.homogenized.txt -p union.c2.cp.pid.reactome.v7.2.symbols.gmt -o Tumor
```
