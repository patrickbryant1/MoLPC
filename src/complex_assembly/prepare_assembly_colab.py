import argparse
import sys
import os
from collections import defaultdict
import numpy as np
import glob
import pandas as pd
import itertools
import shutil
import pdb

################Functions################

#Rewrite functions
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

def read_all_chains_coords(pdbname):
    '''Get all atom coordinates for all chains
    '''

    with open(pdbname) as pdbfile:
        pdb_chains = {} #Coordinates
        prev_res_no=''
        prev_atm = ''
        for line in pdbfile:
            if not line.startswith('ATOM'):
                continue
            record = parse_atm_record(line)
            if record['atm_name']==prev_atm and record['res_no']==prev_res_no:
                continue
            if record['chain'] in [*pdb_chains.keys()]:
                pdb_chains[record['chain']].append(line)
                prev_res_no= record['res_no']
                prev_atm = record['atm_name']
            else:
                pdb_chains[record['chain']] = [line]
                prev_res_no= record['res_no']
                prev_atm = record['atm_name']


    return pdb_chains


def write_pdb(chains, outname):
    '''Save the CB coordinates for later processing
    '''

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    with open(outname,'w') as file:
        #Write chains
        ci=0 #chain name index
        chain = chains['A'] #Only one chain from FD

        prev_res=0
        #Write the chains
        for line in chain:
            record = parse_atm_record(line)
            #Update chain name index
            if record['res_no']>prev_res+200:
                ci+=1
            chain_name = alphabet[ci]
            outline = line[:21]+chain_name+line[22:]
            file.write(outline)
            prev_res=record['res_no']


#Copy pred functions
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



#Rewrite AF PDB
def write_pdb_chain_labels(chains, chain_names, outname):
    '''Save the CB coordinates for later processing
    '''


    with open(outname,'w') as file:
        #Write chains
        ci=0 #chain name index
        for key in chains:
            chain = chains[key]
            chain_name = chain_names[ci]
            #Write the first pdb file
            for line in chain:
                record = parse_atm_record(line)
                outline = line[:21]+chain_name+line[22:]
                file.write(outline)

            #Update chain name index
            ci+=1


#Write all pairs
def read_pdb_for_pairs(pdbfile):
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

def write_pdb_for_pairs(content,pdbfile):
    '''Write dimer to a file
    '''
    with open(pdbfile, 'w') as file:
        for line in content:
            file.write(line)

def get_all_pairs(pdbdir, pairdir, interactions, get_all, meta_df_name):
    '''Get and write all possible pairs
    '''
    #Get all complex files
    subcomplexes = glob.glob(pdbdir+'*.pdb')
    meta_df = {'Source':[], 'Chain1':[], 'Chain2':[]}
    for sub in subcomplexes:
        #Get name
        source_name = sub.split('/')[-1][:-4]
        #Read
        pdb_chains, chain_coords = read_pdb_for_pairs(sub)

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
                    write_pdb_for_pairs(np.append(chi_content, chj_content), pairdir+source_name+'_'+chi+'-'+source_name+'_'+chj+'.pdb')
                    #Save
                    meta_df['Source'].append(source_name)
                    meta_df['Chain1'].append(chi)
                    meta_df['Chain2'].append(chj)

    #Save meta
    meta_df = pd.DataFrame.from_dict(meta_df)
    meta_df.to_csv(meta_df_name, index=None)
    print('Written pairs to', pairdir)
