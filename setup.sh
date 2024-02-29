## Install python packages
conda env create -f molpc.yml
wait
conda activate molpc
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install dm-haiku
## Download AF2 parameters
wait
mkdir ./src/AF2/params
cd ./src/AF2/params
wget https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar
tar -xf alphafold_params_2021-07-14.tar
rm alphafold_params_2021-07-14.tar

## Download uniclust30_2018_08
wait
cd ../../../data/
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz
tar -zxvf uniclust30_2018_08_hhsuite.tar.gz
rm uniclust30_2018_08_hhsuite.tar.gz

## Install HHblits
wait
cd ../src
mkdir hh-suite
cd hh-suite
wget https://github.com/soedinglab/hh-suite/releases/download/v3.3.0/hhsuite-3.3.0-SSE2-Linux.tar.gz
tar xvfz hhsuite-3.3.0-SSE2-Linux.tar.gz
cd ../..
