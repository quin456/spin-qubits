


import sys
from pdb import set_trace


fp = sys.argv[1]+'/'


f = open(fp+'pbs_test', 'w')


f.write('#!/bin/bash\n')
f.write('# -P na4')
f.write('#PBS -q normal\n')
f.write('#PBS -l walltime=20:00:00\n')
f.write('#PBS -l mem=48GB\n')
f.write('#PBS -l jobfs=1GB\n')
f.write('#PBS -l ncpus=12\n')
f.write('#PBS -l wd\n\n')

f.write('module load openmpi/3.0.4\n\n')

f.write('ulimit -s unlimited\n\n')

f.write('mpirun /scratch/na4/qa4681/nemo_1P_2P/'+fp+'nemo3d.ex silicon4_P.xml > my_output.out')