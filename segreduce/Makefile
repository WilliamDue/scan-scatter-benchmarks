DATA_PATH=../data
COMMON_PATH=../common
FUTHARK_PROGRAM=futhark_segreduce
CUDA_PROGRAM=cuda_segreduce
COMPILER?=nvcc
FLAGS?=-O3 --std=c++14 -diag-suppress 550 -diag-suppress 20054
RED=\u001b[31m
GREEN=\u001b[32m
DEFAULT=\033[39m

default: bench

.PHONY: clean bench

$(DATA_PATH)/randomints_sparse_500MiB.in:
	(cd $(DATA_PATH) && make)

$(DATA_PATH)/randomints_dense_500MiB.in:
	(cd $(DATA_PATH) && make)

$(DATA_PATH)/randomints_moderate_500MiB.in:
	(cd $(DATA_PATH) && make)

$(DATA_PATH)/randomints_empty_500MiB.in:
	(cd $(DATA_PATH) && make)

$(DATA_PATH)/randomints_full_500MiB.in:
	(cd $(DATA_PATH) && make)

$(FUTHARK_PROGRAM): $(FUTHARK_PROGRAM).fut
	futhark pkg sync
	futhark cuda $< -o $(FUTHARK_PROGRAM) --server

randomints_sparse_500MiB.flags: $(DATA_PATH)/randomints_sparse_500MiB.in $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'flags ($$loaddata "$<")' -b >$@

randomints_dense_500MiB.flags: $(DATA_PATH)/randomints_dense_500MiB.in $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'flags ($$loaddata "$<")' -b >$@

randomints_moderate_500MiB.flags: $(DATA_PATH)/randomints_moderate_500MiB.in $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'flags ($$loaddata "$<")' -b >$@

randomints_empty_500MiB.flags: $(DATA_PATH)/randomints_empty_500MiB.in $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'flags ($$loaddata "$<")' -b >$@

randomints_full_500MiB.flags: $(DATA_PATH)/randomints_full_500MiB.in $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'flags ($$loaddata "$<")' -b >$@

randomints_sparse_500MiB.out: $(DATA_PATH)/randomints_sparse_500MiB.in randomints_sparse_500MiB.flags $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'expected ($$loaddata "$(DATA_PATH)/randomints_sparse_500MiB.in") ($$loaddata "randomints_sparse_500MiB.flags")' -b >$@

randomints_dense_500MiB.out: $(DATA_PATH)/randomints_dense_500MiB.in randomints_dense_500MiB.flags $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'expected ($$loaddata "$(DATA_PATH)/randomints_dense_500MiB.in") ($$loaddata "randomints_dense_500MiB.flags")' -b >$@

randomints_moderate_500MiB.out: $(DATA_PATH)/randomints_moderate_500MiB.in randomints_moderate_500MiB.flags $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'expected ($$loaddata "$(DATA_PATH)/randomints_moderate_500MiB.in") ($$loaddata "randomints_moderate_500MiB.flags")' -b >$@

randomints_empty_500MiB.out: $(DATA_PATH)/randomints_empty_500MiB.in randomints_empty_500MiB.flags $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'expected ($$loaddata "$(DATA_PATH)/randomints_empty_500MiB.in") ($$loaddata "randomints_empty_500MiB.flags")' -b >$@

randomints_full_500MiB.out: $(DATA_PATH)/randomints_full_500MiB.in randomints_full_500MiB.flags $(FUTHARK_PROGRAM)
	futhark script ./$(FUTHARK_PROGRAM) 'expected ($$loaddata "$(DATA_PATH)/randomints_full_500MiB.in") ($$loaddata "randomints_full_500MiB.flags")' -b >$@

$(CUDA_PROGRAM): $(CUDA_PROGRAM).cu $(COMMON_PATH)/sps.cu.h $(COMMON_PATH)/util.cu.h $(COMMON_PATH)/data.h
	$(COMPILER) $(FLAGS) -o $@ $<

bench: $(FUTHARK_PROGRAM) \
      $(DATA_PATH)/randomints_dense_500MiB.in \
      $(DATA_PATH)/randomints_sparse_500MiB.in \
      $(DATA_PATH)/randomints_moderate_500MiB.in \
      $(DATA_PATH)/randomints_empty_500MiB.in \
      $(DATA_PATH)/randomints_full_500MiB.in \
      randomints_dense_500MiB.flags \
      randomints_sparse_500MiB.flags \
      randomints_moderate_500MiB.flags \
      randomints_empty_500MiB.flags \
      randomints_full_500MiB.flags \
      randomints_dense_500MiB.out \
      randomints_sparse_500MiB.out \
      randomints_moderate_500MiB.out \
      randomints_empty_500MiB.out \
      randomints_full_500MiB.out \
      $(CUDA_PROGRAM)
	@echo -e "$(RED)=== FUTHARK SEGREDUCE ===$(DEFAULT)"
	@futhark bench $(FUTHARK_PROGRAM).fut --backend=cuda --skip-compilation
	@echo -e "$(RED)===============$(DEFAULT)"
	@echo -e "$(GREEN)=== CUDA SEGREDUCE ===$(DEFAULT)"
	@./$(CUDA_PROGRAM) $(DATA_PATH)/randomints_full_500MiB.in     randomints_full_500MiB.flags     randomints_full_500MiB.out
	@echo ""
	@./$(CUDA_PROGRAM) $(DATA_PATH)/randomints_dense_500MiB.in    randomints_dense_500MiB.flags    randomints_dense_500MiB.out
	@echo ""
	@./$(CUDA_PROGRAM) $(DATA_PATH)/randomints_moderate_500MiB.in randomints_moderate_500MiB.flags randomints_moderate_500MiB.out
	@echo ""
	@./$(CUDA_PROGRAM) $(DATA_PATH)/randomints_sparse_500MiB.in   randomints_sparse_500MiB.flags   randomints_sparse_500MiB.out
	@echo ""
	@./$(CUDA_PROGRAM) $(DATA_PATH)/randomints_empty_500MiB.in    randomints_empty_500MiB.flags    randomints_empty_500MiB.out
	@echo -e "$(GREEN)============$(DEFAULT)"

clean:
	rm -rf $(CUDA_PROGRAM) $(FUTHARK_PROGRAM) *.c *.out *.flags *.c $(FUTHARK_PROGRAM)