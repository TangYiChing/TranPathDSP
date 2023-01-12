"""
"""


import os
import sys
import argparse
import numpy as np
import pandas as pd
import gseapy as gp
from datetime import datetime

def retrieveSSGSEA(expMat, geneGmt, tmpStr):
    """
    return pathway-level gene expression matrix

    :param expMat: matrix of expression with cell by gene
    :param geneGmt: list of pathway genes in gmt format
    :param tmpStr: string representing prefix for temporary folder
    :return ssgseaMat: matrix of expression with cell by pathway

    NOTE:
    ======

    Reference:
    https://gseapy.readthedocs.io/en/master/gseapy_tutorial.html#prepare-an-tabular-text-file-of-gene-expression-like-this
    """
    # load data
    gct = expMat.T # gene*cell 
    gmt = geneGmt
    # create temporary folder
    tmp_str = tmpStr +'_ssgsea/'
    try:
        os.mkdir(tmp_str)
    except OSError as error:
        print(error)
        sys.exit(1)
    # run enrichment
    ssgsea = gp.ssgsea(data=gct,  #gct: a matrix of gene by sample
                           gene_sets=gmt, #gmt format
                           outdir=tmp_str,
                           scale=True,
                           permutation_num=2, #1000
                           no_plot=True,
                           processes=10,
                           #min_size=0,
                           format='png')
    # return pathway-level gene expression matrix
    result_mat = ssgsea.res2d.T # get the normalized enrichment score (i.e., NES)
    return result_mat

def parse_parameter():
    parser = argparse.ArgumentParser(description = "Return expression-ssGSEA dataframe")

    parser.add_argument("-e", "--expression_path",
                        required = True,
                        help = "path to expression file, cell by gene matrix")
    parser.add_argument("-p", "--pathway_path",
                        required = True,
                        help = "path to pathway file, in gmt format")
    parser.add_argument("-o", "--output_path",
                        required = True,
                        help = "path to output cell by pathway matrix")
    parser.add_argument("-debug", "--DEBUG",
                        default = True,
                        type = bool,
                        help = "display print message if True")
    return parser.parse_args()

if __name__ == "__main__":
    datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'
    start_time = datetime.now()

    # get args
    args = parse_parameter()
    # load data
    exp_df = pd.read_csv(args.expression_path, header=0, index_col=0, sep="\t")
    # perform ssGSEA
    exp_pathway_df = retrieveSSGSEA(exp_df, args.pathway_path, args.output_path)
    # save to file
    exp_pathway_df.to_csv(args.output_path+'.ssGSEA.txt', header=True, index=True, sep="\t")

    spend = datetime.strptime(str(datetime.now()), datetimeFormat) - datetime.strptime(str(start_time),datetimeFormat)
    print('[Finished in {:}]'.format(spend))
