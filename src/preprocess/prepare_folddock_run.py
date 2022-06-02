import argparse
import sys
import os
import pandas as pd
import numpy as np
import itertools
import shutil
import glob
import pdb

from pair_and_block import read_a3m, match_top_species, write_a3m

parser = argparse.ArgumentParser(description = '''Create the folder structure for all MSAs to run the prediction using FoldDock.''')

parser.add_argument('--msadir', nargs=1, type= str, default=sys.stdin, help = 'Path to data.')
parser.add_argument('--complex_id', nargs=1, type= str, default=sys.stdin, help = 'Id of complex used to make the MSA and output directories.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory (where to create the folder structure). Include /in end')

parser.add_argument('--useqs', nargs=1, type= str, default=sys.stdin, help = 'Path to sequences and stoichiometry for each chain in the complex.')
parser.add_argument('--interactions', nargs='?', const='', type= str, default=sys.stdin, help = 'Known interactions between chains for the complex. If not present, all possible are used.')
parser.add_argument('--intchain2seq', nargs=1, type= str, default=sys.stdin, help = 'Mapping between each chain in the known interactions to the sequences in meta.')
parser.add_argument('--get_all', nargs=1, type= int, default=sys.stdin, help = 'If to get all interactions of size subsize.')
parser.add_argument('--subsize', nargs=1, type= int, default=sys.stdin, help = 'Size of the smallest sub-component to predict (e.g. predict all dimers (2), trimers (3) or entire complex (1)).')


#########Functions###########

def get_sub_combos(uints, subsize):
    '''Get all possible combinations of size subsize
    according to the uints
    '''
    uints_subsize = []
    uints_subsize_check = []
    for chain in np.unique(uints.values):
        sel = uints[(uints['Useq_x']==chain)|(uints['Useq_y']==chain)]
        sel = sel.reset_index()
        #Check that there are more than one int
        if len(sel)<2:
            #Save homo-trimer
            uints_subsize.append(np.array([chain,chain,chain]))
            continue
        for i in range(len(sel)):
            rowi = sel.loc[i]
            rowi_chains = rowi.values[1:]
            for j in range(i+1,len(sel)):
                rowj = sel.loc[j]
                rowj_chains = rowj.values[1:]
                #Get combos
                combos = [x for x in itertools.combinations(np.concatenate([rowi_chains,rowj_chains]), subsize)]
                for combo in combos:
                    if '-'.join([str(x) for x in combo]) not in uints_subsize_check:
                        uints_subsize.append(combo)
                        uints_subsize_check.append('-'.join([str(x) for x in combo]))
    
    return np.array(uints_subsize)


def create_msas(msadir, complex_id, combo, outdir):
    '''Pair and block msas and write them to outdir
    '''
    #Get data
    a3mfiles = []
    a3mspecies = []
    x,y = [],[]
    for chain in combo:
        a3m, species = read_a3m(msadir+complex_id+'_'+str(chain)+'.a3m')
        a3mfiles.append(a3m)
        a3mspecies.append(species)
        x.append(a3m.shape[0]) #Shape
        y.append(a3m.shape[1])


    #Construct block a3m matrix
    #Construct entire a3m matrix
    blocked = np.zeros((np.sum(x),np.sum(y)))
    blocked[:]=21 #Assign gaps
    x_sofar=0
    y_sofar=0
    for i in range(len(a3mfiles)):
        #Assign a3m to blocked
        blocked[x_sofar:x_sofar+a3mfiles[i].shape[0],y_sofar:y_sofar+a3mfiles[i].shape[1]]=a3mfiles[i]
        x_sofar+=a3mfiles[i].shape[0]
        y_sofar+=a3mfiles[i].shape[1]

        #Pair msas
        if i>0:
            paired, matching_species = match_top_species(species1, a3mspecies[i], a3m1, a3mfiles[i])
            species1, a3m1 = matching_species, paired
        else:
            species1=a3mspecies[0]
            a3m1=a3mfiles[0]


    #Write the blocked MSA
    write_a3m(blocked, outdir+complex_id+'_'+'-'.join([str(x) for x in combo])+'_blocked.a3m')
    #Write the paired MSA
    write_a3m(paired, outdir+complex_id+'_'+'-'.join([str(x) for x in combo])+'_paired.a3m')

def create_folder_structure(msadir, complex_id, outdir, useqs, interactions, intchain2seq, get_all, subsize):
    '''Create the folder structure for FoldDock
    '''


    #Check if interactions are present
    if (len(interactions)>0 and get_all==False):
        #Merge the interactions to the intchain2seq to get the unique interactions
        interactions = pd.merge(interactions,intchain2seq,left_on='Chain1',right_on='Chain',how='left')
        interactions = pd.merge(interactions,intchain2seq,left_on='Chain2',right_on='Chain',how='left')
        uints = interactions[['Useq_x','Useq_y']].drop_duplicates()
        #Get the combos
        if subsize>2:
            combos = get_sub_combos(uints, subsize)
        else:
            combos = uints.values

    else:
        #Get all possible combinations of size subsize
        print('Creating all interactions of size',subsize,'...')

        #Create all combinations
        chains = []
        for ind,row in useqs.iterrows():
            chains.extend([row.SeqID]*row.Stoichiometry)
        combos = [x for x in itertools.combinations(chains, subsize)]
        #Get unique combos
        ucombos = []
        for combo in combos:
            if combo not in ucombos:
                ucombos.append(combo)
        combos = ucombos

    #Go through all combinations and create the FoldDock MSAs
    for combo in combos:
        combo = np.sort(combo)
        #Check if the paired MSA has been written
        if os.path.exists(outdir+complex_id+'_'+'-'.join([str(x) for x in combo])+'_paired.a3m'):
            continue

        #Write fasta
        with open(outdir+complex_id+'_'+'-'.join([str(x) for x in combo])+'.fasta', 'w') as file:
            file.write('>'+complex_id+'_'+'-'.join([str(x) for x in combo]))
            seqlens = [len(useqs[useqs.SeqID==chain]['Sequence'].values[0]) for chain in combo]
            file.write('|'+'-'.join([str(x) for x in seqlens])+'\n')
            for chain in combo:
                file.write(useqs[useqs.SeqID==chain]['Sequence'].values[0])

        #Pair, block and write MSAs
        create_msas(msadir, complex_id, combo, outdir)


################MAIN###############
#Parse args
args = parser.parse_args()
#Data
msadir = args.msadir[0]
complex_id = args.complex_id[0]
outdir = args.outdir[0]
useqs = pd.read_csv(args.useqs[0])
#Check if interactions - and get them
if args.interactions:
    interactions = pd.read_csv(args.interactions)
else:
    interactions = ''
intchain2seq = pd.read_csv(args.intchain2seq[0])
get_all = bool(args.get_all[0])
subsize = args.subsize[0]

if subsize>3:
    print('Currently only support subsizes up to 3')
    sys.exit()

#Create the folder structure for the AFM run
create_folder_structure(msadir, complex_id, outdir, useqs, interactions, intchain2seq, get_all, subsize)
