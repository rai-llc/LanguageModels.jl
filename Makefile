.PHONY: clean

models/stories15M/stories15M.bin:
	make -C models/stories15M

clean:
	make -C models/stories15M clean
