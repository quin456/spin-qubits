

module unload intel-compiler intel-mkl python python2 python3 hdf5

module load python3/3.9.2
module load hdf5/1.10.5

python3 -m pip install -v --user ray
