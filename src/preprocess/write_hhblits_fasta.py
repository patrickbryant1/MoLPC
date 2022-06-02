import argparse
import sys
import os
import numpy as np
import pandas as pd
import pdb

parser = argparse.ArgumentParser(description = '''Fetch all unique chains and write their sequences to fasta files.''')
parser.add_argument('--unique_seq_df', nargs=1, type= str, default=sys.stdin, help = 'Path to csv with unique sequences for the complex')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory')

###################FUNCTIONS###################

def write_fasta(df, outdir):
    '''Assign chain ids for each sequence in each complex
    '''

    for ind,row in df.iterrows():
        #Write fasta for MSA creation
        with open(outdir+row['Entry ID']+'_'+str(row.SeqID)+'.fasta', 'w') as file:
            file.write('>'+row['Entry ID']+'_'+str(row.SeqID)+'\n')
            file.write(row.Sequence+'\n')


#################MAIN####################

#Parse args
args = parser.parse_args()
#Args
unique_seq_df = pd.read_csv(args.unique_seq_df[0])
outdir = args.outdir[0]
write_fasta(unique_seq_df, outdir)
