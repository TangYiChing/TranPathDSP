# processed drug structure -- CHEM

```{python}

# given SMILE string, generate molecular fingerprint with size of 256 bits
# note: RDKit is required
$ python ./script/smile2bit.py ./raw_data/DB.Drug.Smile.txt -o Drug
```

# processed drug-target associated network -- DGNet

```{python}

# given gene targets, generate pathway enrichment score 
# note: please install the NetPEA tools at: https://github.com/TangYiChing/NetPEA
$ python ./NetPEA/NetPEA/run_netpea.py -r ./raw_data/DB.Drug.Target.txt -ppi ./STRING/9606.protein_name.links.v11.0.pkl -p ./MSigDB/union.c2.cp.pid.reactome.v7.2.symbols.gmt -o Drug
```

# processed gene expression -- EXP

```{python}
# 1. given gene expression, generate harmonized expression values
# note: please install pycombat at: https://epigenelabs.github.io/pyComBat/
$ python ./run_combat.py -exp ./raw_data/GDSC.expression.txt ./raw_data/Lee2021.expression.txt ./raw_data/GeoSearch.expression.txt ./raw_data/Ding2016.expression.txt -anno ./processed_data/RESP/GDSC.resp.1-AUC.alias.txt ./processed_data/RESP/Lee2021.resp.Alias.txt ./processed_data/RESP/GeoSearch.resp.Alias.txt Ding2016.resp.Alias.txt -o Tumor

# 2. given harmonized expression, generate pathway enrichment score
# note: please install ssGSEA at: https://gseapy.readthedocs.io/en/latest/introduction.html
$ python ./run_ssGSEA.py -exp Tumor.combat.homogenized.txt -p ./MSigDB/union.c2.cp.pid.reactome.v7.2.symbols.gmt -o Tumor
```
