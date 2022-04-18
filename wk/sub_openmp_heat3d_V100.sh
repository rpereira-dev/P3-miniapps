#!/bin/bash
#PBS -q sg8
#PBS -l select=1:ncpus=12:mpiprocs=1:ompthreads=12:ngpus=1
#PBS -l walltime=01:00:00
#PBS -P CityLBM@PG22010

cd $PBS_O_WORKDIR

module load cuda/11.0 nvidia/22.1

#../build/vlp4d/stdpar/vlp4d SLD10_large.dat
../build/heat3D/openmp/heat3D
