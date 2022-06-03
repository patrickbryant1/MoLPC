import sys
import os
import pandas as pd
import numpy as np
import itertools
import shutil
import glob

#########Functions###########
def write_fasta(df, outdir):
    '''Assign chain ids for each sequence in each complex
    '''

    for ind,row in df.iterrows():
        #Write fasta for MSA creation
        with open(outdir+row['Entry ID']+'_'+str(row.SeqID)+'.fasta', 'w') as file:
            file.write('>'+row['Entry ID']+'_'+str(row.SeqID)+'\n')
            file.write(row.Sequence+'\n')

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
