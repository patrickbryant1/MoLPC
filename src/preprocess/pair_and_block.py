import argparse
import sys
import os
import numpy as np
import pandas as pd
import glob
import pdb

# parser = argparse.ArgumentParser(description = '''Bloack and pair the MSAs for interacting chains.''')
#
# parser.add_argument('--a3m1', nargs=1, type= str, default=sys.stdin, help = 'Path to a3m file 1.')
# parser.add_argument('--a3m2', nargs=1, type= str, default=sys.stdin, help = 'Path to a3m file 2.')
# parser.add_argument('--max_gap_fraction', nargs=1, type=float, default=sys.stdin, help = 'The maximal gap fraction allowed in each sequence (default = 0.9).')
# parser.add_argument('--outname', nargs=1, type= str, default=sys.stdin, help = 'Path to file to write to. The suffix _blocked.a3m and _paired.a3m will be added for each respective case.')

def read_a3m(infile,max_gap_fraction=0.9):
    '''Read a3m MSA'''
    mapping = {'-': 21, 'A': 1, 'B': 21, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
             'G': 6,'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,'N': 12,
             'O': 21, 'P': 13,'Q': 14, 'R': 15, 'S': 16, 'T': 17,
             'V': 18, 'W': 19, 'Y': 20,'U': 21, 'Z': 21, 'X': 21, 'J': 21}

    parsed = []#Save extracted msa
    species = []
    seqlen = 0
    lc = 0
    with open(infile, 'r') as file:
        for line in file:
            line = line.rstrip()

            if line.startswith('>'): #species=OrganismIdentifier
                if 'OX=' in line:
                    OX= line.split('OX=')[1]
                    if len(OX)>0:
                        species.append(int(OX.split(' ')[0]))
                    else:
                        species.append(0)
                else:
                    species.append(0)
                continue
            line = line.rstrip()
            gap_fraction = line.count('-') / float(len(line))
            if gap_fraction <= max_gap_fraction:#Only use the lines with less than 90 % gaps
                parsed.append([mapping.get(ch, 22) for ch in line if not ch.islower()])
            else:
                if len(species)>1:
                    species = species[:-1] #Remove the previously stored species
                    continue
            #Check that the lengths match
            if len(parsed[-1])!=seqlen and lc>=1:
                parsed = parsed[:-1]
                species = species[:-1]
                continue
            seqlen = len(parsed[-1])
            lc+=1


    return np.array(parsed, dtype=np.int8, order='F'), np.array(species)

def match_top_species(species1, species2, msa1, msa2):
    '''Select the top species match (first match) in each MSA and merge
    the sequences to a final MSA file in a3m format
    '''
    #Don't remove the zeros (no species), then the query sequences (first line)
    #will be removed
    matching_species = np.intersect1d(species1,species2)

    ind1 = [] #Index to select from the individual MSAs
    ind2 = []
    ncombos = []
    #Go through all matching and select the first (top) hit
    for species in matching_species:
        ind1.append(min(np.argwhere(species1==species)[:,0]))
        ind2.append(min(np.argwhere(species2==species)[:,0]))

        ncombos.append(np.argwhere(species1==species).shape[0]*np.argwhere(species2==species).shape[0])

    #Select from MSAs and merge
    merged = np.concatenate((msa1[ind1], msa2[ind2]),axis=1)

    return merged, matching_species

def write_a3m(fused, outfile):
    '''Write a3m MSA'''
    backmap = { 1:'A', 2:'C', 3:'D', 4:'E', 5:'F',6:'G' ,7:'H',
               8:'I', 9:'K', 10:'L', 11:'M', 12:'N', 13:'P',14:'Q',
               15:'R', 16:'S', 17:'T', 18:'V', 19:'W', 20:'Y', 21:'-'} #Here all unusual AAs and gaps are set to the same char (same in the GaussDCA script)

    with open(outfile,'w') as file:
        for i in range(len(fused)):
            file.write('>'+str(i)+'\n')
            file.write(''.join([backmap[ch] for ch in fused[i]])+'\n')

    return None


