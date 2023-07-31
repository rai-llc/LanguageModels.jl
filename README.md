# LanguageModels.jl

A port of
[@karpathy](https://github.com/karpathy)'s
[llama2.c](https://github.com/karpathy/llama2.c)
to pure Julia.

[![Build Status](https://github.com/rai-llc/LanguageModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/rai-llc/LanguageModels.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Special licensing exception

This repo contains one external file, [sentencepiece_model.proto](https://github.com/google/sentencepiece/blob/635fe8423a249b6e081aacd290d8aef7476c6a28/src/sentencepiece_model.proto), which has its own [copyright and license](https://github.com/google/sentencepiece/blob/635fe8423a249b6e081aacd290d8aef7476c6a28/LICENSE).
This file is used to generate a Julia interface in the [sentencepiece](https://github.com/rai-llc/LanguageModels.jl/tree/main/src/sentencepiece] subdirectory,
which is then used to load the tokenizer model for llama2.

## Basic usage

```julia-repl
julia> using LanguageModels

julia> LanguageModels.talkto(temperature=0.0)
REPL mode language_model_mode initialized. Press ~ to enter and backspace to exit.
"Prompt(\"language_model> \",...)"

language_model> Once upon a time, there was a llama named
[ Info: dim = 288
[ Info: hidden_dim = 768
[ Info: n_layers = 6
[ Info: n_heads = 6
[ Info: n_kv_heads = 6
[ Info: vocab_size = 32000
[ Info: seq_len = 256
[ Info: shared_weights = true
[ Info: vocab_size = 32000
Once upon a time, there was a llama named Joe. Joe was a very happy little boy who loved to play. One day, Joe was playing in the park when he saw a big, red balloon. Joe was so excited and he wanted to play with it.
Joe ran over to the balloon and started to play with it. He was having so much fun, he didn't notice the balloon was getting bigger and bigger. Suddenly, the balloon popped and Joe was so sad.
Joe's mom saw what happened and she said, "Joe, why did you poke the balloon? You should be more careful." Joe felt very sorry and said, "I'm sorry, Mommy. I didn't mean to poke the balloon."
Joe's mom hugged him and said, "It's okay, Joe. We all make mistakes. But remember, it's important to be careful and think before you do something. That way, you can avoid accidents and stay safe."
Joe nodded and said, "I will remember, Mommy. I will be more careful next time."
```

Supported keyword arguments:
- `checkpoint_filename`: defaults to stories15M artifact which is downloaded when package is built
- `tokenizer_filename`: ditto
- `temperature::Float32`: how much randomness to use in sampling. Default: `0.9f0`
- `steps`: How many tokens to generate. Default: 256. Must not exceed `seq_len` in the model definition.
- `stop_on_special_tokens`. Stop sampling tokens once the special tokens such as UNK (unknown), BOS (beginning-of-sentence) or EOS (end of sentence) are encountered. Default: `true`. For behavior similar to `llama2.c`, set to `false`.

## Models

This implementation by default uses the `stories15M` model from Andrej Karpathy's tinyllamas project on HuggingFace, and its corresponding `tokenizer.bin` from the `llama2.c` repo.
This model is automatically downloaded when the package is built.

### llama-2
If you have access to the [llama-2](https://huggingface.co/meta-llama) model weights,
you can follow the [llama2.c instructions](https://github.com/karpathy/llama2.c#metas-llama-2-models)
for converting them into a usable format.
The `tokenizer.model` can be read as is. (The code here automatically generates the reader format using the sentencepiece [Protobuf specification](https://github.com/google/sentencepiece/blob/635fe8423a249b6e081aacd290d8aef7476c6a28/src/sentencepiece_model.proto).)


```jl
julia> LanguageModels.main(checkpoint_filename="/Users/jiahao/models/llama/llama-2-7b.bin", tokenizer_filename="/Users/jiahao/models/llama/tokenizer.model", tokenizer_loader=LanguageModels.load_sentencepiece_model, prompt="Once upon a time, there was a llama called",)
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