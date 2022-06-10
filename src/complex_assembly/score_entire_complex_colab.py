import argparse
import sys
import os
import numpy as np
import pandas as pd
import glob
from collections import defaultdict
import pdb


################FUNCTIONS#################
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
    chain_CA_inds = {}
    chain_CB_inds = {}
    chain_plddt = {}

    with open(pdbfile) as file:
        for line in file:
            record = parse_atm_record(line)
            if record['chain'] in [*pdb_chains.keys()]:
                pdb_chains[record['chain']].append(line)
                chain_coords[record['chain']].append([record['x'],record['y'],record['z']])
                coord_ind+=1
                if record['atm_name']=='CA':
                    chain_CA_inds[record['chain']].append(coord_ind)
                    chain_plddt[record['chain']].append(record['B'])
                if record['atm_name']=='CB' or (record['atm_name']=='CA' and record['res_name']=='GLY'):
                    chain_CB_inds[record['chain']].append(coord_ind)


            else:
                pdb_chains[record['chain']] = [line]
                chain_coords[record['chain']]= [[record['x'],record['y'],record['z']]]
                chain_CA_inds[record['chain']]= []
                chain_CB_inds[record['chain']]= []
                chain_plddt[record['chain']]= []
                #Reset coord ind
                coord_ind = 0


    return pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds, chain_plddt

def score_complex(path_coords, path_CB_inds, path_plddt):
    '''Score all interfaces in the current complex
    '''
    metrics = {'Chain':[], 'n_ints':[], 'sum_av_IF_plDDT':[],
                'n_contacts':[], 'n_IF_residues':[]}

    chains = [*path_coords.keys()]
    chain_inds = np.arange(len(chains))
    #Get interfaces per chain
    for i in chain_inds:
        chain_i = chains[i]
        chain_coords = np.array(path_coords[chain_i])
        chain_CB_inds = path_CB_inds[chain_i]
        l1 = len(chain_CB_inds)
        chain_CB_coords = chain_coords[chain_CB_inds]
        chain_plddt = np.array(path_plddt[chain_i])
        #Metrics
        n_chain_ints = 0
        chain_av_IF_plDDT = 0
        n_chain_contacts = 0
        n_chain_IF_residues = 0

        for int_i in np.setdiff1d(chain_inds, i):
            int_chain = chains[int_i]
            int_chain_CB_coords = np.array(path_coords[int_chain])[path_CB_inds[int_chain]]
            int_chain_plddt = np.array(path_plddt[int_chain])
            #Calc 2-norm
            mat = np.append(chain_CB_coords,int_chain_CB_coords,axis=0)
            a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
            dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
            contact_dists = dists[:l1,l1:]
            contacts = np.argwhere(contact_dists<=8)
            #The first axis contains the contacts from chain 1
            #The second the contacts from chain 2
            if contacts.shape[0]>0:
                n_chain_ints += 1
                chain_av_IF_plDDT +=  np.concatenate((chain_plddt[contacts[:,0]], int_chain_plddt[contacts[:,1]])).mean()
                n_chain_contacts += contacts.shape[0]
                n_chain_IF_residues += np.unique(contacts).shape[0]

        #Save
        metrics['Chain'].append(chain_i)
        metrics['n_ints'].append(n_chain_ints)
        metrics['sum_av_IF_plDDT'].append(chain_av_IF_plDDT) #Divide with n_ints to get avg per int
        metrics['n_contacts'].append(n_chain_contacts)
        metrics['n_IF_residues'].append(n_chain_IF_residues)
    #Create df
    metrics_df = pd.DataFrame.from_dict(metrics)
    return metrics_df


def calc_mpDockQ(metrics_df):
    '''Calculats the multiple interface pDockQ
    '''

    def sigmoid(x, L ,x0, k, b):
        y = L / (1 + np.exp(-k*(x-x0)))+b
        return y


    av_IF_plDDT = np.average(metrics_df.sum_av_IF_plDDT/metrics_df.n_ints)
    n_contacts= metrics_df.n_contacts.sum()

    L = 0.783
    x0= 289.79
    k= 0.061
    b= 0.23
    mpDockQ = sigmoid(av_IF_plDDT*np.log10(n_contacts+0.001), L ,x0, k, b)

    return mpDockQ

#################MAIN####################

def main(model_id, model, model_path, useqs, chain_seqs, outname):
    #Get all chain lengths
    useqs['Chain_length'] = [len(x) for x in useqs.Sequence]
    useqs = useqs[['SeqID', 'Chain_length']]
    chain_lens = pd.merge(chain_seqs, useqs, left_on='Useq', right_on='SeqID', how='left')
    chain_lens = dict(zip(chain_lens.Chain.values, chain_lens.Chain_length.values))
    #Read PDB
    pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds, chain_plddt = read_pdb(model)
    #Score
    metrics_df = score_complex(chain_coords, chain_CB_inds, chain_plddt)
    #Add id
    metrics_df['ID']=model_id
    #Calc mpDockQ
    mpDockQ = calc_mpDockQ(metrics_df)
    metrics_df['mpDockQ']=mpDockQ
    metrics_df.to_csv(outname, index=None)
    print('mpDockQ:',mpDockQ)
