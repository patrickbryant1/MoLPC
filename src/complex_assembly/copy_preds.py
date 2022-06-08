from collections import defaultdict
import sys
import os
import pandas as pd
import numpy as np
import itertools
import shutil
import glob


parser = argparse.ArgumentParser(description = '''Copy the predictions of all unique interactions to reflect all possible (redundant) interactions.''')

parser.add_argument('--complex_id', nargs=1, type= str, default=sys.stdin, help = 'Id of complex used to make the MSA and output directories.')
parser.add_argument('--pdbdir', nargs=1, type= str, default=sys.stdin, help = 'Path to directory with predictions. Include /in end')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to directory of where the preds should be copied. Include /in end')

parser.add_argument('--useqs', nargs=1, type= str, default=sys.stdin, help = 'Path to sequences and stoichiometry for each chain in the complex.')
parser.add_argument('--subsize', nargs=1, type= int, default=sys.stdin, help = 'Size of the smallest sub-component to predict (e.g. predict all dimers (2), trimers (3) or entire complex (1)).')
parser.add_argument('--intchain2seq', nargs=1, type= str, default=sys.stdin, help = 'Mapping between each chain in the known interactions to the sequences in meta.')
parser.add_argument('--interactions', nargs='?', type= str, default=sys.stdin, help = 'Known interactions between chains for the complex. If not known, specify get all as 1 (True)')
parser.add_argument('--get_all', nargs=1, type= int, default=sys.stdin, help = 'If to get all interactions of size subsize.')


#########Functions###########

def get_combos(chains, subsize):
    '''Get all combinations of chains of a given subsize
    '''
    #Save combos
    combos = []

    if subsize==2:
        chains_i = chains[0]
        chains_j = chains[1]
        #Go through all chains and create all non-repeating instances of 2
        for i in range(len(chains_i)):
            ci = chains_i[i]
            for j in range(len(chains_j)):
                cj = chains_j[j]
                if ci==cj: #Don't save identical chains
                    continue
                else:
                    combos.append([ci,cj])
    if subsize==3:
        chains_i = chains[0]
        chains_j = chains[1]
        chains_k = chains[2]
        #Go through all chains and create all non-repeating instances of 3
        for i in range(len(chains_i)):
            ci = chains_i[i]
            for j in range(len(chains_j)):
                cj = chains_j[j]
                if ci==cj: #Don't save identical chains
                    continue
                for k in range(len(chains_k)):
                    ck = chains_k[k]
                    if ci==ck or cj==ck:
                        continue
                    else:
                        combos.append([ci,cj,ck])

    return combos

def copy_uints(complex_id, pdbdir, outdir, useqs, interactions, intchain2seq, get_all, subsize):
    '''Create the folder structure for AF-multimer
    For each type,
    If numeric --> skip n-1 letters in folder assignment
    E.g. A2B --> make folder A, skip B, make folder C
    '''
     #Make plDDT dir
    if not os.path.exists(outdir+'plddt'):
        os.mkdir(outdir+'plddt')

    #If not get all - map chains according to ints
    if (len(interactions)>0 and get_all==False):
        #Assign chains names according to intchain2seq
        useq2chain = {}
        for useq in intchain2seq.Useq.unique():
            sel = intchain2seq[intchain2seq.Useq==useq]
            useq2chain[useq] = [*sel.Chain.values]
    else:
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        #Assign chains names according to stoichiometry
        useq2chain = {}
        useq2chain_df = {'Chain':[], 'Useq':[]}
        ci=0
        for ind,row in useqs.iterrows():
            uchain = row.SeqID
            useq2chain[uchain]=[]
            for si in range(row.Stoichiometry):
                useq2chain[uchain].append(alphabet[ci])
                useq2chain_df['Chain'].append(alphabet[ci])
                useq2chain_df['Useq'].append(uchain)
                ci+=1

        #Create a new df mapping the chain names to the useqs
        useq2chain_df = pd.DataFrame.from_dict(useq2chain_df)
        #Save
        useq2chain_df.to_csv(outdir+complex_id+'_chains.csv',index=None)

    #Go through all preds and copy to match all requested interactions
    #Do the same with the plDDT
    preds = glob.glob(pdbdir+complex_id+'_*/*_rw.pdb')
    for pred in preds:
        #Get interacting chains
        intchains = pred.split('/')[-2].split('_')[-1].split('-')
        #Get plDDT
        metrics = np.load(glob.glob('/'.join(pred.split('/')[:-1])+'/result_model_1*.pkl')[0],allow_pickle=True)
        plDDT = metrics['plddt']
        #Copy the pred into a new dir for each chain repeat
        chains = []
        for uchain in intchains:
            chains.append(useq2chain[int(uchain)])
        combos = get_combos(chains, subsize)
        #Make dirs, save plDDT and copy
        for combo in combos:
            if not os.path.exists(outdir+complex_id+'_'+''.join(combo)):
                os.mkdir(outdir+complex_id+'_'+''.join(combo))
            #Copy
            shutil.copyfile(pred, outdir+complex_id+'_'+''.join(combo)+'/unrelaxed_model_1_multimer.pdb')
            np.save(outdir+'/plddt/'+complex_id+'_'+''.join(combo)+'.npy',plDDT)
################MAIN###############
#Parse args
args = parser.parse_args()
#Data
complex_id = args.complex_id[0]
pdbdir = args.pdbdir[0]
outdir = args.outdir[0]
useqs = pd.read_csv(args.useqs[0])
subsize = args.subsize[0]
if args.interactions:
    interactions = pd.read_csv(args.interactions)
else:
    interactions=''
intchain2seq = pd.read_csv(args.intchain2seq[0])
get_all = bool(args.get_all[0])

if subsize>3:
    print('Currently only support subsizes up to 3')
    sys.exit()
#Copy the predictions to reflect all chains
copy_uints(complex_id, pdbdir, outdir, useqs,interactions, intchain2seq, get_all, subsize)
