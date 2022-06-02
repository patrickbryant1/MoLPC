import argparse
import sys
import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import glob
import subprocess
from scipy.spatial import distance
from circuit import hamilton, all_paths
import pdb

parser = argparse.ArgumentParser(description = '''Write all interacting pairs.''')

parser.add_argument('--pdbdir', nargs=1, type= str, default=sys.stdin, help = 'Path to data with complexes.')
parser.add_argument('--pairdir', nargs=1, type= str, default=sys.stdin, help = 'Path to where to write parwise complexes.')
parser.add_argument('--meta', nargs=1, type= str, default=sys.stdin, help = 'Path to output csv with all binary interactions')
parser.add_argument('--interactions', nargs='?', type= str, default=sys.stdin, help = 'Known interactions between chains for the complex. If not known, specify get all as 1 (True)')
parser.add_argument('--get_all', nargs=1, type= int, default=sys.stdin, help = 'If to get all interactions of size subsize.')

##############FUNCTIONS###############
def parse_atm_record(line):
    '''Get the atm record
    '''
    record = defaultdict()
    record['name'] = line[0:6].strip()
    record['atm_no'] = int(line[6:11])
    record['atm_name'] = line[12:16].strip()
    record['atm_alt'] = line[17]
    record['res_name'] = line[17:20].strip()
    record['chain'] = line[21]
    record['res_no'] = int(line[22:26])
    record['insert'] = line[26].strip()
    record['resid'] = line[22:29]
    record['x'] = float(line[30:38])
    record['y'] = float(line[38:46])
    record['z'] = float(line[46:54])
    record['occ'] = float(line[54:60])
    record['B'] = float(line[60:66])

    return record

def read_pdb(pdbfile):
    '''Read a pdb file per chain
    '''
    pdb_chains = {}
    chain_coords = {}
    with open(pdbfile) as file:
        for line in file:
            record = parse_atm_record(line)
            if record['chain'] in [*pdb_chains.keys()]:
                pdb_chains[record['chain']].append(line)
                #Get CB (CA for GLY)
                if record['atm_name']=='CB' or (record['atm_name']=='CA' and record['res_name']=='GLY'):
                    chain_coords[record['chain']].append([record['x'],record['y'],record['z']])

            else:
                pdb_chains[record['chain']] = [line]
                chain_coords[record['chain']] = []


    return pdb_chains, chain_coords

def write_pdb(content,pdbfile):
    '''Write dimer to a file
    '''
    with open(pdbfile, 'w') as file:
        for line in content:
            file.write(line)

################MAIN###############
#Parse args
args = parser.parse_args()
#Data
pdbdir = args.pdbdir[0]
pairdir = args.pairdir[0]
if args.interactions:
    interactions = pd.read_csv(args.interactions)
else:
    interactions = ''
get_all = bool(args.get_all[0])

#Get all complex files
subcomplexes = glob.glob(pdbdir+'*.pdb')
meta_df = {'Source':[], 'Chain1':[], 'Chain2':[]}
for sub in subcomplexes:
    #Get name
    source_name = sub.split('/')[-1][:-4]
    #Read
    pdb_chains, chain_coords = read_pdb(sub)

    #Analyse all pairs
    sub_chains = [*pdb_chains.keys()]
    for i in range(len(sub_chains)-1):
        chi = sub_chains[i]
        chi_coords = chain_coords[chi]
        l1 = len(chi_coords)
        chi_content = pdb_chains[chi]
        for j in range(i+1,len(sub_chains)):
            chj = sub_chains[j]
            chj_coords = chain_coords[chj]
            chj_content = pdb_chains[chj]
            #Check if get all or to use known interactions
            if get_all==False:
                sel = interactions[(interactions.Chain1==chi)&(interactions.Chain2==chj)]
                if len(sel)<1:
                    continue
            #Calculate contacts
            #Calc 2-norm
            mat = np.append(chi_coords,chj_coords,axis=0)
            a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
            dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
            contact_dists = dists[l1:,:l1]
            contacts = np.argwhere(contact_dists<=8)
            #Write joint file if contacts
            if contacts.shape[0]>0:
                write_pdb(np.append(chi_content, chj_content), pairdir+source_name+'_'+chi+'-'+source_name+'_'+chj+'.pdb')
                #Save
                meta_df['Source'].append(source_name)
                meta_df['Chain1'].append(chi)
                meta_df['Chain2'].append(chj)

#Save meta
meta_df = pd.DataFrame.from_dict(meta_df)
meta_df.to_csv(args.meta[0], index=None)
print('Written pairs to', pairdir)
