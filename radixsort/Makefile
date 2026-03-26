DATA_PATH=../data
COMMON_PATH=../common
FUTHARK_PROGRAM=futhark_radixsort
CUDA_PROGRAM=cuda_radixsort
COMPILER?=nvcc
FLAGS?=-O3 --std=c++14 -diag-suppress 550
RED=\u001b[31m
GREEN=\u001b[32m
DEFAULT=\033[39m

default: bench

.PHONY: clean bench

$(DATA_PATH)/randomints_moderate_500MiB.in:
	(cd $(DATA_PATH) && make)

$(FUTHARK_PROGRAM): $(FUTHARK_PROGRAM).fut
	futhark cuda --server $< -o $(FUTHARK_PROGRAM)

randomints_moderate_500MiB.out: $(DATA_PATH)/randomints_moderate_500MiB.in
	futhark script ./$(FUTHARK_PROGRAM) 'expected ($$loaddata "$<")' -b >$@

bench: $(CUDA_PROGRAM) \
	   $(FUTHARK_PROGRAM) \
	   randomints_moderate_500MiB.out \
	   $(DATA_PATH)/randomints_moderate_500MiB.in
	@echo -e "$(RED)=== FUTHARK FILTER ===$(DEFAULT)"
	@futhark bench $(FUTHARK_PROGRAM).fut --backend=cuda
	@echo -e "$(RED)===============$(DEFAULT)"
	@echo -e "$(GREEN)=== CUDA FILTER ===$(DEFAULT)"
	@./$(CUDA_PROGRAM) $(DATA_PATH)/randomints_moderate_500MiB.in ./randomints_moderate_500MiB.out
	@echo -e "$(GREEN)============$(DEFAULT)"

$(CUDA_PROGRAM): $(CUDA_PROGRAM).cu $(COMMON_PATH)/sps.cu.h $(COMMON_PATH)/util.cu.h $(COMMON_PATH)/data.h
	$(COMPILER) $(FLAGS) -o $@ $<

clean:
	rm -rf $(CUDA_PROGRAM) $(FUTHARK_PROGRAM) *.c *.out *.c $(FUTHARK_PROGRAM)
