CFLAGS=-fopenmp-offload-mandatory -foffload-lto
CFLAGS+=-O3
CFLAGS+=-DENABLE_OPENMP_OFFLOAD
CFLAGS+=-I ../ext_lib/mdspan/include/ -I ../miniapps/heat3d/ -I ../miniapps/heat3d/openmp/ -I ../lib/openmp/
SRC=../miniapps/heat3d/energy.cc ../miniapps/heat3d/openmp/heat3d.cpp

ARGS=--nx 512 --ny 512 --nz 512 --nbiter 10000 --freq_diag 0
ENV=OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES="cores(8)"

aurora:
	icpx $(CFLAGS) $(SRC) -fiopenmp -fopenmp-targets=spir64_gen -Xopenmp-target-backend "-device pvc" -DZES_ENERGY=1 -lze_loader -I /usr/include/level_zero/
	$(ENV) ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE ./a.out $(ARGS)
	$(ENV) ZE_FLAT_DEVICE_HIERARCHY=FLAT      ./a.out $(ARGS)

nvidia:
	clang++ $(CFLAGS) $(SRC) -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target=nvptx64 --offload-arch=sm_70,sm_80,sm_90 -lcuda -lcudart -lnvidia-ml -DCUDA_ENERGY=1
	$(ENV) ./a.out $(ARGS)

amd:
	clang++ $(CFLAGS) $(SRC) -I $(ROCM_PATH)/include -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa --offload-arch=gfx90a,gfx942 -DROCM_ENERGY=1 -L $(ROCM_PATH)/lib/ -lamdhip64 -L $(ROCM_PATH)/lib/ -lrocm_smi64
	$(ENV) ./a.out $(ARGS)
