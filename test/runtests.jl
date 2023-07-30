using Downloads
using TransformerREPL
using Test

@testset "TransformerREPL.jl" begin

    # Initial tests to replicate llama2.c behavior
    model_file = Downloads.download("https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin")
    tokenizer_file = Downloads.download("https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin")

    buf = IOBuffer
    main(model_file, tokenizer_file, temperature = 0.0f0, steps = 256, prompt = "", io=buf)
    @test String(take!(buf)) == """Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
    Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to knock the ball down.
    Lily found a stick and tried to hit the ball. But the stick was too short. She tried again and again, but she couldn't reach it. She felt sad.
    Suddenly, a kind man came by and saw Lily. He asked her what was wrong. Lily told him about the ball. The man smiled and said, "I have a useful idea!" He took out a long stick and used it to knock the ball down. Lily was so happy! She thanked the man and they played together in the sunshine."""

    main("stories15M.bin", "tokenizer.bin", temperature = 0.0f0, steps = 256, prompt = "Once upon a time, there was a dog", io=buf)
    @test String(take!(buf)) == """Once upon a time, there was a dog named Max. Max was a very happy dog and he loved to play. One day, Max was playing in the park when he saw a big, scary cat. Max was so scared that he started to run away.
    Max ran and ran until he was very tired. He stopped to take a rest and then he saw a big, scary cat. Max was so scared that he started to cry.
    The cat said, "Don't be scared, little dog. I won't hurt you. I just want to be your friend."
    Max was still scared, but he was also very brave. He said, "Okay, I'm sorry. I won't be scared anymore."
    The cat smiled and said, "That's okay. I'm not scary. I just want to be friends."
    Max was so happy that he had made a new friend. He and the cat played together all day and had lots of fun."""

end
