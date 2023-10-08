# LanguageModels.jl

A port of
[@karpathy](https://github.com/karpathy)'s
[llama2.c](https://github.com/karpathy/llama2.c)
to pure Julia.

[![Build Status](https://github.com/rai-llc/LanguageModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/rai-llc/LanguageModels.jl/actions/workflows/CI.yml?query=branch%3Amain)

[![Video of LanguageModels.jl running llama2-13b](https://img.youtube.com/vi/EXMTGwwpPV8/sddefault.jpg)](https://youtu.be/EXMTGwwpPV8)

## Special licensing exception

This repo contains one external file, [sentencepiece_model.proto](https://github.com/google/sentencepiece/blob/635fe8423a249b6e081aacd290d8aef7476c6a28/src/sentencepiece_model.proto), which has its own [copyright and license](https://github.com/google/sentencepiece/blob/635fe8423a249b6e081aacd290d8aef7476c6a28/LICENSE).
This file is used to generate a Julia interface in the [sentencepiece](https://github.com/rai-llc/LanguageModels.jl/tree/main/src/sentencepiece) subdirectory,
which is then used to load the tokenizer model for llama2.

## How to install

This package is currently unregistered.
To install it, run

```jl
julia> using Pkg; Pkg.add(url="https://github.com/rai-llc/LanguageModels.jl")
```

## Basic usage

```jl
julia> using LanguageModels
REPL mode language_model_mode initialized. Press ~ to enter and backspace to exit.

julia> LanguageModels.repl_kwargs[:temperature] = 0.0
0.0

language_model> Once upon a time, there was a llama named Johnny
[ Info: Temperature: 0.0
[ Info: Steps: 256
[ Info: dim = 288
[ Info: hidden_dim = 768
[ Info: n_layers = 6
[ Info: n_heads = 6
[ Info: n_kv_heads = 6
[ Info: seq_len = 256
[ Info: shared_weights = true
[ Info: vocab_size = 32000
One day, Johnny was playing with his friends when they all had to go to the store. Johnny was very excited to show them the way.
When they arrived, Johnny's noticed that Johnny was carrying a lot of news. He was so happy that he started to run around the store, but he was too fast.
Suddenly, Johnny stopped and looked around. He saw a big box with a sign on it. He knew that Johnny was going to the store, but he was too excited to
[ Info: achieved tok/s: 550.5533719471397
```

Supported keyword arguments:
- `checkpoint_filename`: defaults to stories15M artifact which is downloaded when package is built
- `tokenizer_filename`: ditto.
   The `tokenizer.model` file should be in the `sentencepiece`
  [`ModelProto` Protocol Buffers format](https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto).
- `temperature::Float32`: how much randomness to use in sampling. Default: `0.9f0`
- `steps`: How many tokens to generate. Default: 256. Must not exceed `seq_len` in the model definition.
- `mmap`: Whether to use [memory mapped I/O](https://docs.julialang.org/en/v1/stdlib/Mmap/#Mmap.mmap)
   to load models that are too large to fit in memory. Default: `false`
- `stop_on_special_tokens`. Stop sampling tokens once the special tokens such as UNK (unknown), BOS (beginning-of-sentence) or EOS (end of sentence) are encountered. Default: `true`. For behavior similar to `llama2.c`, set to `false`.

## Models

This implementation by default uses the `stories15M` model from Andrej Karpathy's tinyllamas project on HuggingFace, and its corresponding `tokenizer.model` from the `llama2.c` repo.
This model is automatically downloaded when the package is built.

### llama-2
If you have access to the [llama-2](https://huggingface.co/meta-llama) model weights,
you can follow the [llama2.c instructions](https://github.com/karpathy/llama2.c#metas-llama-2-models)
for converting them into a usable format.
The `tokenizer.model` can be read as is. (The code here automatically generates the reader format using the sentencepiece [Protobuf specification](https://github.com/google/sentencepiece/blob/635fe8423a249b6e081aacd290d8aef7476c6a28/src/sentencepiece_model.proto).)


```jl
julia> LanguageModels.main(checkpoint_filename="/Users/jiahao/models/llama/llama-2-7b.bin", tokenizer_filename="/Users/jiahao/models/llama/tokenizer.model", prompt="Once upon a time, there was a llama called",)
Once upon a time, there was a llama called Pinky and a sheep called Blue.
Pinky was a llama who ate vermicelli. He was very fussy about how it was cooked. He wanted it long, and hot and slightly damp. It was the only way he would eat it.
He lived with two humans, Flora and Henry. They didn’t like him very much. They found him funny-looking and they thought he was smelly.
Flora thought he smelled of durian.
One day, Blue and Pinky went to their favourite restaurant together. It was called Not Very Delicious at All.
I’ll leave it you to make up your own mind. You can hear the poem performed here.
The background I used was the free pixel artist at the top of the sidebar.
Great story. I need to fin a pic of just flora, or just Henry. Do you have one or do I need to make one?
I don’t have one with Flora alone. I’ll have a look through my stash and see if I can find one though.
```
