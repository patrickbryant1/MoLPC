
#############PARAMETERS#############
BASE=$(pwd) #Where all scripts are run from, now the current directory
ID=1A8R
DATADIR=$BASE/data/test/
HHBLITS=$BASE/hhblits #Path to hhblits version 3.1.0
HHBLITSDB=$BASE/data/uniclust30_2018_08/uniclust30_2018_08
###Note!
#This runscript assumes that you have sinfularity in your path

#Run the assembly pipeline starting from predicted interactions
#########INPUTS and PATHS##########
USEQS=$DATADIR/$ID'_useqs.csv'
CHAINS=$DATADIR/$ID'_chains.csv'
INTERACTIONS='' #Leave empty if the interactions are not known - here they are not used. See the file $DATADIR/$ID'_ints.csv' to how to supply such a file
SUBSIZE=3 #What order the subcomplexes should be (2 or 3)
GET_ALL=1 #If to get all interactions (1) or not (0) - when the interactions are known
#########OUTPUTS#########
#The best assembled complex ,the assembly path and its score (mpDockQ)

#########Step1: MSA PIPELINE#########
SINGIMG=$BASE/src/AF2/AF_environment.sif #Sing img
HHBLITSDB=$BASE/data/uniclust30_2018_08/uniclust30_2018_08
MSADIR=$DATADIR/hhblits
mkdir $MSADIR
#Write individual fasta files for all unique sequences
singularity exec $SINGIMG python3 $BASE/src/preprocess/write_hhblits_fasta.py --unique_seq_df $USEQS \
--outdir $MSADIR/

#Run HHblits
for file in $MSADIR/*.fasta
do
  SUBID=$(echo $file|cut -d '/' -f 6|cut -d '.' -f 1)
  singularity exec $SINGIMG hhblits -i $file -d $HHBLITSDB -E 0.001 -all -n 2 -oa3m $MSADIR/$SUBID'.a3m'
done

#########Step2: FOLDING PIPELINE#########
wait
#Write the Paired and Block Diagonalized MSAs to predict sub-components
singularity exec $SINGIMG python3 $BASE/src/preprocess/all_pdb_over9/prepare_folddock_run.py --msadir $MSADIR/ \
--complex_id $ID \
--outdir $MSADIR/ \
--useqs $USEQS \
--interactions $INTERACTIONS \
--intchain2seq $CHAINS \
--get_all $GET_ALL \
--subsize $SUBSIZE

#Create structure dir
STRUCTURE_DIR=$DATADIR/AF
mkdir $STRUCTURE_DIR
#Get subids
head -n 1 $MSADIR/*.fasta|grep '|'|cut -d '|' -f 1|cut -d '>' -f 2 > $DATADIR/sub_ids.txt
#Get lens for AF run
head -n 1 $MSADIR/*.fasta|grep '|'|cut -d '|' -f 2|cut -d '-' -f 1-2 > $DATADIR/lens.txt

#Predict the subcomponents
##### AF2 CONFIGURATION #####
PARAM=$BASE'/src/AF2/'
PRESET='full_dbs' #Choose preset model configuration - no ensembling (full_dbs) and (reduced_dbs) or 8 model ensemblings (casp14).
MAX_RECYCLES=10 #max_recycles (default=3)
MODEL_NAME='model_1' #model_1_ptm

#Go through all subcomponents and predict their structure
NCOMPONENTS=$(wc -l < $DATADIR/sub_ids.txt)
for ((LN=1;LN<=NCOMPONENTS;LN++))
do
  SUBID=$(sed -n $LN'p' $DATADIR/sub_ids.txt)
  echo $SUBID
  ####Get fasta file####
  FASTAFILE=$MSADIR/$SUBID'.fasta'
  ####Get chain break#### Note! This is now set for trimer subcomponents
  CB=$(sed -n $LN'p' $DATADIR/lens.txt)
  CB1=$(echo $CB|cut -d '-' -f 1)
  CB2=$(echo $CB|cut -d '-' -f 2)
  CB2=$(( $CB1 + $CB2 ))
  CB=$CB1,$CB2
  ####Get MSAs####
  #HHblits paired
  PAIREDMSA=$MSADIR/$SUBID'_paired.a3m'
  ##HHblits block diagonalized
  BLOCKEDMSA=$MSADIR/$SUBID'_blocked.a3m'
  MSAS="$PAIREDMSA,$BLOCKEDMSA" #Comma separated list of msa paths
  singularity exec --nv $SINGIMG python3 $BASE/src/AF2/run_alphafold.py \
                --fasta_paths=$FASTAFILE \
                --msas=$MSAS \
                --chain_break_list=$CB \
                --output_dir=$STRUCTURE_DIR \
                --model_names=$MODEL_NAME \
                --data_dir=$PARAM \
                --fold_only \
                --uniref90_database_path=$HHBLITSDB \
                --mgnify_database_path=$HHBLITSDB \
                --bfd_database_path=$HHBLITSDB \
                --uniclust30_database_path=$HHBLITSDB \
                --pdb70_database_path=$HHBLITSDB \
                --template_mmcif_dir=$HHBLITSDB \
                --obsolete_pdbs_path=$HHBLITSDB \
                --preset=$PRESET \
                --max_recycles=$MAX_RECYCLES

done

#########Step3: ASSEMBLY PIPELINE#########
wait
COMPLEXDIR=$DATADIR/assembly/complex/ #Where all the output for the complex assembly will be directed
CODEDIR=$BASE/src/complex_assembly
#Make complex directory
mkdir -p $COMPLEXDIR
#Rewrite the FoldDock preds to have separate chains according to the fasta file seqlens
singularity exec $SINGIMG python3 $CODEDIR/rewrite_fd.py --pdbdir $STRUCTURE_DIR --pdb_id $ID

#Copy all predicted unique chain interactions to reflect all possible interactions
SUB_PDBDIR=$STRUCTURE_DIR/
OUTDIR=$DATADIR/assembly/
singularity exec $SINGIMG python3 $CODEDIR/copy_preds.py --complex_id $ID --pdbdir $SUB_PDBDIR --outdir $OUTDIR \
--useqs $USEQS --subsize $SUBSIZE --interactions $INTERACTIONS --intchain2seq $CHAINS --get_all $GET_ALL

#Rewrite AF predicted complexes to have proper numbering and chain labels
PDBDIR=$OUTDIR
singularity exec $SINGIMG python3 $CODEDIR/rewrite_af_pdb.py --pdbdir $PDBDIR --pdb_id $ID --outdir $OUTDIR

#Write all pairs
PAIRDIR=$PDBDIR/pairs/
META=$PDBDIR/meta.csv #where to write all interactions
#It is necessary that the first unique chain is named A-..N for and the second N-... and so on
mkdir $PAIRDIR
#Glob for all files with each chain in order (A,B,C,D) A-->B,C,D; B--> C,D; C-->D
singularity exec $SINGIMG python3 $CODEDIR/write_all_pairs.py --pdbdir $PDBDIR --pairdir $PAIRDIR --meta $META \
--interactions $INTERACTIONS --get_all $GET_ALL

#Assemble from pairs
#Find the best non-overlapping path that connect all nodes using Monte Carlo Tree search
PLDDTDIR=$PDBDIR/plddt/
CHAIN_SEQS=$PDBDIR/$ID'_chains.csv' #Updated chain seqs
singularity exec $SINGIMG python3 $CODEDIR/mcts.py --network $META \
--pairdir $PAIRDIR --plddt_dir $PLDDTDIR \
--useqs $USEQS --chain_seqs $CHAIN_SEQS \
--outdir $COMPLEXDIR

#########Step4: SCORING#########
#Score complex
MODEL=$COMPLEXDIR/best_complex.pdb
MODEL_PATH=$COMPLEXDIR/optimal_paths.csv
DT=8
OUTNAME=$COMPLEXDIR/$ID'_score.csv'
singularity exec $SINGIMG python3 $CODEDIR/score_entire_complex.py --model_id $ID --model $MODEL \
--model_path $MODEL_PATH --plddtdir $PLDDTDIR \
--useqs $USEQS --chain_seqs $CHAIN_SEQS \
--outname $OUTNAME
