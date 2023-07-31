# LanguageModelsLanguageModels.talkto(steps=512, temperature=0.0)

A port of
[@karpathy](https://github.com/karpathy)'s
[llama2.c](https://github.com/karpathy/llama2.c)
to pure Julia.

[![Build Status](https://github.com/rai-llc/LanguageModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/rai-llc/LanguageModels.jl/actions/workflows/CI.yml?query=branch%3Amain)

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
- `stop_on_bos`. Stop sampling tokens once the BOS (beginning-of-sentence) marker is encountered. Default: `true`. For behavior similar to `llama2.c`, set to `false`.
```


## Models

This implementation is currently hard-coded to use the `stories15M` model from Andrej Karpathy's tinyllamas project on HuggingFace, and its corresponding `tokenizer.bin` from the `llama2.c` repo.

In the root directory of this repo, you can download these automatically by running

```sh
make
``` 

Then in Julia, run:

```julia-repl
julia> using LanguageModels; LanguageModels.main("stories15M.bin", "tokenizer.bin", temperature = 0.0f0, steps = 256, prompt = "")
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to knock the ball down.
Lily found a stick and tried to hit the ball. But the stick was too short. She tried again and again, but she couldn't reach it. She felt sad.
Suddenly, a kind man came by and saw Lily. He asked her what was wrong. Lily told him about the ball. The man smiled and said, "I have a useful idea!" He took out a long stick and used it to knock the ball down. Lily was so happy! She thanked the man and they played together in the sunshine.
```

```julia-repl
julia> LanguageModels.main("stories15M.bin", "tokenizer.bin", temperature = 0.0f0, steps = 256, prompt = "Once upon a time, there was a dog")
Once upon a time, there was a dog named Max. Max was a very happy dog and he loved to play. One day, Max was playing in the park when he saw a big, scary cat. Max was so scared that he started to run away.
Max ran and ran until he was very tired. He stopped to take a rest and then he saw a big, scary cat. Max was so scared that he started to cry.
The cat said, "Don't be scared, little dog. I won't hurt you. I just want to be your friend."
Max was still scared, but he was also very brave. He said, "Okay, I'm sorry. I won't be scared anymore."
The cat smiled and said, "That's okay. I'm not scary. I just want to be friends."
Max was so happy that he had made a new friend. He and the cat played together all day and had lots of fun.
````
