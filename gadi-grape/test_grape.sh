#PBS -P na4
#PBS -q gpuvolta
#PBS -l ncpus=24
#PBS -l ngpus=2
#PBS -l mem=16GB
#PBS -l jobfs=0GB
#PBS -l walltime=000:05:00
#PBS -l wd
 
# Load module, always specify version number.
module load intel-mkl/2020.3.304
module load python3/3.9.2
module load cuda/11.4.1
module load cudnn/8.2.2-cuda11.4
module load nccl/2.10.3-cuda11.4
module load openmpi/4.1.2
module load magma/2.6.0
module load fftw3/3.3.8
module load pytorch/1.10.0
module use ~/.local/lib/python3.9/site-packages/joblib
# Set number of OMP threads
export OMP_NUM_THREADS=$PBS_NCPUS
 
# Must include `#PBS -l storage=scratch/ab12+gdata/yz98` if the job
# needs access to `/scratch/ab12/` and `/g/data/yz98/`. Details on
# https://opus.nci.org.au/display/Help/PBS+Directives+Explained.
 
# Run Python applications
python3 test_grape.py > test_grape.stdout.$PBS_JOBID
