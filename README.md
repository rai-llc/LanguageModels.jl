# TransformerREPL

A port of
[@karpathy](https://github.com/karpathy))'s
[llama2.c](https://github.com/karpathy/llama2.c)
to pure Julia.

[![Build Status](https://github.com/rai-llc/TransformerREPL.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/rai-llc/TransformerREPL.jl/actions/workflows/CI.yml?query=branch%3Amain)

## A simple example

You will need two files: the `stories15M` model on HuggingFace and its corresponding `tokenizer.bin` from the `llama2.c` repo.

In the root directory of this repo, you can download these automatically by running

```sh
make
``` 

Then in Julia, run:

```julia-repl
julia> using TransformerREPL; TransformerREPL.main("stories15M.bin", "tokenizer.bin", temperature = 0.0f0, steps = 256, prompt = "")
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to knock the ball down.
Lily found a stick and tried to hit the ball. But the stick was too short. She tried again and again, but she couldn't reach it. She felt sad.
Suddenly, a kind man came by and saw Lily. He asked her what was wrong. Lily told him about the ball. The man smiled and said, "I have a useful idea!" He took out a long stick and used it to knock the ball down. Lily was so happy! She thanked the man and they played together in the sunshine.
```

```julia-repl
julia> TransformerREPL.main("stories15M.bin", "tokenizer.bin", temperature = 0.0f0, steps = 256, prompt = "Once upon a time, there was a dog")
Once upon a time, there was a dog named Max. Max was a very happy dog and he loved to play. One day, Max was playing in the park when he saw a big, scary cat. Max was so scared that he started to run away.
Max ran and ran until he was very tired. He stopped to take a rest and then he saw a big, scary cat. Max was so scared that he started to cry.
The cat said, "Don't be scared, little dog. I won't hurt you. I just want to be your friend."
Max was still scared, but he was also very brave. He said, "Okay, I'm sorry. I won't be scared anymore."
The cat smiled and said, "That's okay. I'm not scary. I just want to be friends."
Max was so happy that he had made a new friend. He and the cat played together all day and had lots of fun.
````
