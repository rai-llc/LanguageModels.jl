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



stories15M_path = artifact"stories15M"

"""
    main(;
        checkpoint_filename,
        tokenizer_filename,
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

The defaults for checkpoint_filename and tokenizer_filename load the
stories15M.bin model from Andrej Karpathy's tinyllamas project.
"""
function main(;
        checkpoint_filename = joinpath(stories15M_path, "stories15M.bin"),
        tokenizer_filename = joinpath(stories15M_path, "tokenizer.bin"),
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
