import argparse
import sys
import os
from collections import defaultdict
import numpy as np
import glob
import pdb

parser = argparse.ArgumentParser(description = '''Rewrite the AF2 output to contain two different chains in the PDB files.''')

parser.add_argument('--pdbdir', nargs=1, type= str, default=sys.stdin, help = 'Path to data.')
parser.add_argument('--pdb_id', nargs=1, type= str, default=sys.stdin, help = 'PDB id.')

################Functions################
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


################MAIN###############
#Parse args
args = parser.parse_args()
#Data
pdbdir = args.pdbdir[0]
pdb_id = args.pdb_id[0]

#Read an rewrite all pdb files

#Check if file is present
files = glob.glob(pdbdir+'/'+pdb_id+'*/*1.pdb')
if len(files)>0:
    for pdbname in files:
        chains = read_all_chains_coords(pdbname)
        if len(chains.keys())>1:
            continue
        subid = pdbname.split('/')[-2]
        #Rewrite the files
        write_pdb(chains, pdbname.split('.')[0]+'_rw'+'.pdb')
