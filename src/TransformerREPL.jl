module TransformerREPL

using LinearAlgebra
using LogExpFunctions
using StatsBase

struct Config
    dim::Int        # transformer dimension
    hidden_dim::Int # for ffn layers
    n_layers::Int   # number of layers
    n_heads::Int    # number of query heads
    n_kv_heads::Int # number of key/value heads (can be < query heads because of multiquery)
    vocab_size::Int # vocabulary size, usually 256 (byte-level)
    seq_len::Int    # max sequence length
    shared_weights::Bool
end

function Base.read(io::IOStream, ::Type{Config})
    dim = read(io, Int32)
    @info "dim = $dim"
    hidden_dim = read(io, Int32)
    @info "hidden_dim = $hidden_dim"
    n_layers = read(io, Int32)
    @info "n_layers = $n_layers"
    n_heads = read(io, Int32)
    @info "n_heads = $n_heads"
    n_kv_heads = read(io, Int32)
    @info "n_kv_heads = $n_kv_heads"
    vocab_size = read(io, Int32)
    @info "vocab_size = $vocab_size"
    seq_len = read(io, Int32)
    @info "seq_len = $seq_len"
    shared_weights = vocab_size > 0
    @info "shared_weights = $shared_weights"
    vocab_size = abs(vocab_size)
    @info "vocab_size = $vocab_size"
 
    Config(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, shared_weights)
end

@kwdef struct TransformerWeights{T}
    # cjh: Here, and in many other places, the matrices are interpreted as being indexed with the dimensions reversed relative to llama2.c, to respect the row-major storage format of the original C and Python codes.
    # This means that matrices have to be transposed in downstream matrix products

    token_embedding_table::Matrix{T} # (dim, vocab_size)
    # weights for rmsnorms
    rms_att_weight::Matrix{T} # (dim, layer)
    rms_ffn_weight::Matrix{T} # (dim, layer)
    # weights for matmuls
    wq::Array{T,3} # (dim, dim, layer)
    wk::Array{T,3} # (dim, dim, layer)
    wv::Array{T,3} # (dim, dim, layer)
    wo::Array{T,3} # (dim, dim, layer)
    # weights for ffn
    w1::Array{T,3} # (dim, hidden_dim, layer)
    w2::Array{T,3} # (hidden_dim, dim, layer)
    w3::Array{T,3} # (dim, hidden_dim, layer)
    # final rmsnorm
    rms_final_weight::Vector{T} # (dim,)
    # freq_cis for RoPE relative positional embeddings
    freq_cis_real::Matrix{T} # ((dim / n_heads) / 2, seq_len)
    freq_cis_imag::Matrix{T} # ((dim / n_heads) / 2, seq_len)
end

TransformerWeights(p::Config) = TransformerWeights(;
    token_embedding_table = zeros(Float32, p.dim, p.vocab_size),
    rms_att_weight = zeros(Float32, p.dim, p.n_layers),
    rms_ffn_weight = zeros(Float32, p.dim, p.n_layers),
    wq = zeros(Float32, p.dim, p.dim, p.n_layers),
    wk = zeros(Float32, p.dim, p.dim, p.n_layers),
    wv = zeros(Float32, p.dim, p.dim, p.n_layers),
    wo = zeros(Float32, p.dim, p.dim, p.n_layers),
    w1 = zeros(Float32, p.dim, p.hidden_dim, p.n_layers),
    w2 = zeros(Float32, p.hidden_dim, p.dim, p.n_layers),
    w3 = zeros(Float32, p.dim, p.hidden_dim, p.n_layers),
    rms_final_weight = zeros(Float32, p.dim),
    freq_cis_real = zeros(Float32, (p.dim ÷ p.n_heads) ÷ 2, p.seq_len),
    freq_cis_imag = zeros(Float32, (p.dim ÷ p.n_heads) ÷ 2, p.seq_len),
)

function Base.read!(io::IOStream, w::TransformerWeights)
    read!(io, w.token_embedding_table)
    read!(io, w.rms_att_weight)
    read!(io, w.wq)
    read!(io, w.wk)
    read!(io, w.wv)
    read!(io, w.wo)
    read!(io, w.rms_ffn_weight)
    read!(io, w.w1)
    read!(io, w.w2)
    read!(io, w.w3)
    read!(io, w.rms_final_weight)
    read!(io, w.freq_cis_real)
    read!(io, w.freq_cis_imag)
    return w
end

@kwdef struct RunState{T} # current wave of activations
    x::Vector{T}      # activation at current time stamp (dim,)
    xb::Vector{T}     # same, but inside a residual branch (dim,)
    xb2::Vector{T}    # an additional buffer just for convenience (dim,)
    hb::Vector{T}     # buffer for hidden dimension in the ffn (hidden_dim,)
    hb2::Vector{T}    # buffer for hidden dimension in the ffn (hidden_dim,)
    q::Vector{T}      # query (dim,)
    k::Vector{T}      # key (dim,)
    v::Vector{T}      # value (dim,)
    att::Vector{T}    # buffer for scores/attention values (seq_len,)
    logits::Vector{T} # output logits
    # kv cache
    key_cache::Array{T,3}   # (dim, seq_len, layer)
    value_cache::Array{T,3} # (dim, seq_len, layer)
end

RunState(p::Config) = RunState(;
    x = zeros(Float32, p.dim),
    xb = zeros(Float32, p.dim),
    xb2 = zeros(Float32, p.dim),
    hb = zeros(Float32, p.hidden_dim),
    hb2 = zeros(Float32, p.hidden_dim),
    q = zeros(Float32, p.dim),
    k = zeros(Float32, p.dim),
    v = zeros(Float32, p.dim),
    att = zeros(Float32, p.seq_len),
    logits = zeros(Float32, p.vocab_size),
    key_cache = zeros(Float32, p.dim, p.seq_len, p.n_layers),
    value_cache = zeros(Float32, p.dim, p.seq_len, p.n_layers),
)

function rmsnorm!(o, x, weight; ϵ = 1f-5)
    n = length(x)
    ss = 1/√(x⋅x/n + ϵ)
    # normalize and scale
    o .= weight .* (ss .* x)
end

softmax!(x) = LogExpFunctions.softmax!(x,x)

@views function transformer!(token::Int, pos::Int, p::Config, s::RunState, w::TransformerWeights)
    # a few convenience variables
    x = s.x
    dim = p.dim
    hidden_dim = p.hidden_dim
    head_size = dim ÷ p.n_heads

    # copy the token embedding into x
    x[:] = w.token_embedding_table[:, token]

    # pluck out the "pos" row of freq_cis_real and freq_cis_imag
    freq_cis_real_row = w.freq_cis_real[:, pos]
    freq_cis_imag_row = w.freq_cis_imag[:, pos]

    # forward all the layers
    for l in 1:p.n_layers
        # attention rmsnorm
        rmsnorm!(s.xb, x, w.rms_att_weight[:, l])

        # qkv matmuls for this position
        # cjh: Here, and in many other places, the matrix has to be transposed because of row-major vs column-major issues
        mul!(s.q, w.wq[:, :, l]', s.xb)
        mul!(s.k, w.wk[:, :, l]', s.xb)
        mul!(s.v, w.wv[:, :, l]', s.xb)

        # apply RoPE rotation to the q and k vectors for each head
        for h in 1:p.n_heads
            this_head = ((h-1)*head_size+1):(h*head_size)
            # get the q and k vectors for this head
            q = s.q[this_head]
            k = s.k[this_head]
            # rotate q and k by the freq_cis_real and freq_cis_imag
            for i=1:2:head_size
                q0 = q[i]
                q1 = q[i+1]
                k0 = k[i]
                k1 = k[i+1]
                fcr = freq_cis_real_row[(1+i)÷2]
                fci = freq_cis_imag_row[(1+i)÷2]
                q[i]   = q0 * fcr - q1 * fci
                q[i+1] = q0 * fci + q1 * fcr
                k[i]   = k0 * fcr - k1 * fci
                k[i+1] = k0 * fci + k1 * fcr
            end
        end

        # save key,value at this time step (pos) to our kv cache
        s.key_cache[:, pos, l] =  s.k
        s.value_cache[:, pos, l] = s.v

        # multihead attention. iterate over all heads
        for h in 1:p.n_heads
            this_head = ((h-1)*head_size+1):(h*head_size)
            # get the query vector for this head
            q = s.q[this_head]
            # get the key vector for this head and at all timesteps
            k = s.key_cache[this_head, 1:pos, l]
            # calculate the attention score as the dot product of q and k
            mul!(s.att[1:pos], k', q)
            s.att[1:pos] /= √(head_size)            
            # softmax the scores to get attention weights
            softmax!(s.att[1:pos])

            # weighted sum of the values, store back into xb
            mul!(s.xb[this_head], s.value_cache[this_head, 1:pos, l], s.att[1:pos])
        end

        # final matmul to get the output of the attention
        mul!(s.xb2, w.wo[:, :, l]', s.xb)

        # residual connection back into x
        x .+= s.xb2

        # ffn rmsnorm
        rmsnorm!(s.xb, x, w.rms_ffn_weight[:, l])

        # Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        # first calculate self.w1(x) and self.w3(x)
        mul!(s.hb, w.w1[:, :, l]', s.xb)
        mul!(s.hb2, w.w3[:, :, l]', s.xb)

        # F.silu silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        @. s.hb *= s.hb2*logistic(s.hb)
        
        # final matmul to get the output of the ffn
        mul!(s.xb, w.w2[:, :, l]', s.hb)

        # residual connection
        x .+= s.xb
    end

    # final rmsnorm
    rmsnorm!(x, x, w.rms_final_weight)

    # classifier into logits
    mul!(s.logits, w.token_embedding_table', x)
end

# ----------------------------------------------------------------------------
# byte pair encoding (BPE) tokenizer, encodes strings into tokens so we can prompt

"""
    TokenizedString{T, S<:Integer} <: AbstractString

A simple string type that provides an encoding:

    1:typemax(S) -> alphabet::T

# Implementation note

The AbstractString interface is not complete.
To use in any real use case, convert to a `String` first.
"""
struct TokenizedString{T, S<:Integer} <: AbstractString
    tokens::Vector{S}
    alphabet::Vector{T}
end

# Very basic string interface
# TODO fix repr()
Base.ncodeunits(s::TokenizedString) = length(s.tokens)
Base.firstindex(s::TokenizedString) = 1
Base.isvalid(s::TokenizedString, i::Int) = 1 ≤ i ≤ length(s.tokens)
function Base.isvalid(::Type{TokenizedString}, s::TokenizedString)
    n = length(s.alphabet)
    for i in s.tokens
        if i < 1 || i > n
            return false
        end
    end
    return true
end

function Base.iterate(s::TokenizedString, idx::Int=firstindex(s))
    if idx <= ncodeunits(s) 
        return (s.alphabet[s.tokens[idx]], idx+1)
    end
end

function Base.convert(::Type{String}, s::TokenizedString)
    io = IOBuffer()
    for i in s.tokens
        print(io, s.alphabet[i])
    end
    String(take!(io))
end

"""
    DigramEncodingTokenizer{T, S<:Real} <: AbstractString

A diagram encoding tokenizer emitting tokens of type `T`.
A special case would be a byte pair encoder, with `T = Vector{UInt8}`.
The `scores` of eltype `S` are used to determine the most frequent tokens.
In the simplest case, the scores are the token frequencies.

This implementation is intended to be used with `load_tokenizer()` to retrieve predefined `alphabet` and `scores`.

```julia-repl
julia> enc = load_tokenizer("/Users/jiahao/local/src/llama2.c/tokenizer.bin", 32000); enc("Hello world").tokens
2-element Vector{UInt16}:
 0x2af3
 0x0c73
```
"""
struct DigramEncodingTokenizer{T, S<:Real} <: AbstractString
    alphabet::Vector{T}
    scores::Vector{S}
end

"""
    load_tokenizer(filename, vocab_size) -> DigramEncodingTokenizer{String,Float32}

Loads the tokenizer from the binary file format used by nanoGPT.
"""
function load_tokenizer(filename, vocab_size)
    vocab = Vector{Vector{UInt8}}(undef, vocab_size)
    vocab_scores = Vector{Float32}(undef, vocab_size)
    file = open(filename)
    max_token_length = read(file, Int32)
    for i in 1:vocab_size
        vocab_scores[i] = read(file, Float32)
        len = read(file, Int32)
        if len > max_token_length
            @error "Encountered token of length $len exceeding maximum of $max_token_length"
        end
        vocab[i] = read(file, len)
    end
    if !eof(file)
        @warn "Stopped before end of file was reached: $filename"
    end
    close(file)
    
    DigramEncodingTokenizer([String(copy(s)) for s in vocab], vocab_scores)
end

"""
    _infer_int_type(n) -> T <: Integer

Finds the smallest (unsigned) integer type that can represent the positive number `n`.
"""
function _infer_int_type(n)
    if n < typemax(UInt8)
        UInt8
    elseif n < typemax(UInt16)
        UInt16
    elseif n < typemax(UInt32)
        UInt32
    elseif n < typemax(UInt64)
        UInt64
    elseif n < typemax(UInt128)
        UInt128
    else
        BigInt # Should never be needed
    end
end

# call method
function (enc::DigramEncodingTokenizer)(text::String)

    alphabet = enc.alphabet
    scores = enc.scores
    R = _infer_int_type(length(alphabet))
    T = eltype(scores)

    tokens = R[]
    # First encode every character
    for ch in text
        char = string(ch)
        id = findfirst(isequal(char), alphabet)
        if isnothing(id)
            @error "\"$char\" not in alphabet"
        end
        push!(tokens, id)
    end
    
    while true # Keep merging consecutive pairs
        best_score = typemin(T)
        best_id = best_idx = 0

        for i = 1:length(tokens)-1
            # check if we can merge the pair (tokens[i], tokens[i+1])
            token = alphabet[tokens[i]]*alphabet[tokens[i+1]]
            id = findfirst(isequal(token), alphabet)
            if (!isnothing(id) && scores[id] > best_score) 
                # this merge pair exists in alphabet! record its score and position
                best_score = scores[id]
                best_id = id
                best_idx = i
            end
        end
            
        if (best_idx == 0) 
            @debug "Done"
            break # we couldn't find any more pairs to merge, so we're done
        end

        # merge the consecutive pair (best_idx, best_idx+1) into new token best_id
#         a = alphabet[best_idx]
#         b = alphabet[best_idx+1]
#         @debug "Merging ($a, $b)"
        tokens[best_idx] = best_id
        # delete token at position best_idx+1, shift the entire sequence back 1
        deleteat!(tokens, best_idx+1)
    end

    TokenizedString(tokens, alphabet)
end


"""
    main(
        checkpoint_filename,
        tokenizer_filename;
        temperature = 0.9f0,
        steps = 256,
        prompt = "",
        stop_on_bos = true,
        io = stdout)

This implementation has been tested on the stories15M nano GPT model
and mostly reproduces the output of llama2.c at zero temperature with no prompt.
The main difference is that it does not print the starting BOS token,
and terminates the text generation early if a BOS token is encountered
later. To reproduce the exact output of llama2.c, specify the kwarg
stop_on_bos = false.

Two test cases have been tested against llama2.c:

```julia-repl
julia> main("stories15M.bin", "tokenizer.bin", temperature = 0.0f0, steps = 256, prompt = "")
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to knock the ball down.
Lily found a stick and tried to hit the ball. But the stick was too short. She tried again and again, but she couldn't reach it. She felt sad.
Suddenly, a kind man came by and saw Lily. He asked her what was wrong. Lily told him about the ball. The man smiled and said, "I have a useful idea!" He took out a long stick and used it to knock the ball down. Lily was so happy! She thanked the man and they played together in the sunshine.
```

```julia-repl
julia> main("stories15M.bin", "tokenizer.bin", temperature = 0.0f0, steps = 256, prompt = "Once upon a time, there was a dog")
Once upon a time, there was a dog named Max. Max was a very happy dog and he loved to play. One day, Max was playing in the park when he saw a big, scary cat. Max was so scared that he started to run away.
Max ran and ran until he was very tired. He stopped to take a rest and then he saw a big, scary cat. Max was so scared that he started to cry.
The cat said, "Don't be scared, little dog. I won't hurt you. I just want to be your friend."
Max was still scared, but he was also very brave. He said, "Okay, I'm sorry. I won't be scared anymore."
The cat smiled and said, "That's okay. I'm not scary. I just want to be friends."
Max was so happy that he had made a new friend. He and the cat played together all day and had lots of fun.
````
"""
function main(
        checkpoint_filename,
        tokenizer_filename;
        temperature = 0.9f0,
        steps = 256,
        prompt = "",
        stop_on_bos = true,
        io = stdout
    )

    # read in the model.bin file
    config, weights = open(checkpoint_filename) do file
        config = read(file, Config)
        weights = TransformerWeights(config)
        read!(file, weights)
        config, weights
    end

    # read in the tokenizer.bin file
    tokenizer = load_tokenizer(tokenizer_filename, config.vocab_size)
    vocab = tokenizer.alphabet
    state = RunState(config)

    # process the prompt, if any
    s = tokenizer(prompt)
    num_prompt_tokens = length(s)
    prompt_tokens = s.tokens

    # start the main loop
    start_time = nothing# used to time our code, only initialized after first iteration
    token = 2   # init with BOS token, as done in Llama-2 sentencepiece tokenizer
    pos = 1     # position in the sequence
    while (pos <= steps) 
        # forward the transformer to get logits for the next token
        transformer!(token, pos, config, state, weights)

        if pos <= num_prompt_tokens 
            # if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos]
        elseif (temperature == 0.0f0) # sample the next token
        
                # greedy argmax sampling: take the token with the highest probability
                next = argmax(state.logits)
        else # sample the next token at finite temperature
            # apply the temperature to the logits
            state.logits ./= temperature
            # apply softmax to the logits to get the probabilities for next token
            softmax!(state.logits)
            # we sample from this distribution to get the next token
            # p, i = findmax(state.logits)
            next::Int = sample(ProbabilityWeights(state.logits, 1.0f0))
            # error( p, " - ", i, " - ", vocab[i])
        end

        # following BOS token (2), sentencepiece decoder strips any leading whitespace (see PR #89)
        token_str = vocab[next]
        if token == 2
            token_str = lstrip(token_str)
        end

        # cjh: This behavior deviates from llama2.c if stop_on_bos == true
        if stop_on_bos && next == 2
            break
        end

        print(io, token_str)

        # advance forward
        token = next
        pos += 1
        # init our timer here because the first iteration is slow
        if isnothing(start_time)
            start_time = time_ns()
        end
    end
    println()

    # report our achieved tok/s
    speed = config.seq_len / (time_ns() - start_time)*1e9
    @info "achieved tok/s: $speed"

    return nothing
end
end
