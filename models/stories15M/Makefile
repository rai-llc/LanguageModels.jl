.PHONY: all clean

all: stories15M.bin tokenizer.bin

clean:
	rm stories15M.bin tokenizer.bin

stories15M.bin:
	curl -O -L https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

tokenizer.bin:
	curl -O -L https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
