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

function Base.read(io::IO, ::Type{Config})
    dim = read(io, Int32)
    hidden_dim = read(io, Int32)
    n_layers = read(io, Int32)
    n_heads = read(io, Int32)
    n_kv_heads = read(io, Int32)
    vocab_size = read(io, Int32)
    seq_len = read(io, Int32)
    shared_weights = vocab_size > 0
    vocab_size = abs(vocab_size)

    @info "dim = $dim"
    @info "hidden_dim = $hidden_dim"
    @info "n_layers = $n_layers"
    @info "n_heads = $n_heads"
    @info "n_kv_heads = $n_kv_heads"
    @info "seq_len = $seq_len"
    @info "shared_weights = $shared_weights"
    @info "vocab_size = $vocab_size"

    Config(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, shared_weights)
end

@kwdef struct TransformerWeights{T, Vec <: AbstractVector{T}, Mat <: AbstractMatrix{T}, Arr3 <: AbstractArray{T, 3}}
    # cjh: Here, and in many other places, the matrices are interpreted as being indexed with the dimensions reversed relative to llama2.c, to respect the row-major storage format of the original C and Python codes.
    # This means that matrices have to be transposed in downstream matrix products

    config::Config

    token_embedding_table::Mat # (dim, vocab_size)
    # weights for rmsnorms
    rms_att_weight::Mat # (dim, layer)
    rms_ffn_weight::Mat # (dim, layer)
    # weights for matmuls
    wq::Arr3 # (dim, dim, layer)
    wk::Arr3 # (dim, dim, layer)
    wv::Arr3 # (dim, dim, layer)
    wo::Arr3 # (dim, dim, layer)
    # weights for ffn
    w1::Arr3 # (dim, hidden_dim, layer)
    w2::Arr3 # (hidden_dim, dim, layer)
    w3::Arr3 # (dim, hidden_dim, layer)
    # final rmsnorm
    rms_final_weight::Vec # (dim,)
    # freq_cis for RoPE relative positional embeddings
    freq_cis_real::Mat # ((dim / n_heads) / 2, seq_len)
    freq_cis_imag::Mat # ((dim / n_heads) / 2, seq_len)

    wcls::Mat # (dim, vocab_size)
end

function checksizes(w::TransformerWeights)
    dim = w.config.dim
    hidden_dim = w.config.hidden_dim
    vocab_size = w.config.vocab_size
    layer = w.config.n_layers
    n_heads = w.config.n_heads
    seq_len = w.config.seq_len

    @assert size(w.token_embedding_table) == (dim, vocab_size)
    @assert size(w.rms_att_weight) == (dim, layer)
    @assert size(w.rms_ffn_weight) == (dim, layer)
    @assert size(w.wq) == (dim, dim, layer)
    @assert size(w.wk) == (dim, dim, layer)
    @assert size(w.wv) == (dim, dim, layer)
    @assert size(w.wo) == (dim, dim, layer)
    @assert size(w.w1) == (dim, hidden_dim, layer)
    @assert size(w.w2) == (hidden_dim, dim, layer)
    @assert size(w.w3) == (dim, hidden_dim, layer)
    @assert size(w.rms_final_weight) == (dim,)
    @assert size(w.freq_cis_real) == ((dim ÷ n_heads) ÷ 2, seq_len)
    @assert size(w.freq_cis_imag) == ((dim ÷ n_heads) ÷ 2, seq_len)
    @assert size(w.wcls) == (dim, vocab_size)    
end

@views function load_model(checkpoint_filename; materialize=copy)
    v = Mmap.mmap(checkpoint_filename, Vector{Float32})

    # Use the first 7*4=28 bytes to read the config
    p = read(IOBuffer(reinterpret(UInt8, v[1:7])), Config)

    # The remaining bytes are the model weights
    f = v[8:end]
    marker = 1

    load = (n...) -> begin
        step = *(n...)
        out = reshape(f[marker:(marker+step-1)], n...)
        marker += step
        return materialize(out)
    end

    token_embedding_table = load(p.dim, p.vocab_size)
    rms_att_weight = load(p.dim, p.n_layers)
    wq = load(p.dim, p.dim, p.n_layers)
    wk = load(p.dim, p.dim, p.n_layers)
    wv = load(p.dim, p.dim, p.n_layers)
    wo = load(p.dim, p.dim, p.n_layers)
    rms_ffn_weight = load(p.dim, p.n_layers)
    w1 = load(p.dim, p.hidden_dim, p.n_layers)
    w2 = load(p.hidden_dim, p.dim, p.n_layers)
    w3 = load(p.dim, p.hidden_dim, p.n_layers)
    rms_final_weight = load(p.dim)
    freq_cis_real = load((p.dim ÷ p.n_heads) ÷ 2, p.seq_len)
    freq_cis_imag = load((p.dim ÷ p.n_heads) ÷ 2, p.seq_len)

    if p.shared_weights
        wcls = token_embedding_table
    else
        wcls = load(p.dim, p.vocab_size)
    end

    weights = TransformerWeights(; config=p, token_embedding_table, rms_att_weight, rms_ffn_weight, wq, wk, wv,
                                wo, w1, w2, w3, rms_final_weight, freq_cis_real, freq_cis_imag, wcls)
    return p, weights
end


"""
Compute frequencies for rotary positional encoding (RoPE)
"""
function RoPE(dim::Int, last::Int, θ::Real = 10000)
    freqs = (θ .^ -(0.0f0:2.0f0/dim:1.0f0)[1:((dim÷2))])
    t = 0:(last-1)
    freqs_mat = freqs*t'
    freqs_cos = cos.(freqs_mat)  # real part
    freqs_sin = sin.(freqs_mat)  # imaginary part
    return freqs_cos, freqs_sin
end


load_torch_model(checkpoint_filename, parameters_filename) =
    load_torch_model(Array{Float32}, checkpoint_filename, parameters_filename)

@views function load_torch_model(::Type{T},
	checkpoint_filename,
	parameters_filename,
    ) where {S, T <: AbstractArray{S}}

    params_dict = open(parameters_filename) do f
        JSON.parse(read(f, String))
    end

    dim = params_dict["dim"]
    n_heads = params_dict["n_heads"]
    n_kv_heads = get(params_dict, "n_kv_heads", n_heads)
    n_layers = params_dict["n_layers"] 

    vars = Pickle.Torch.THload(checkpoint_filename)

    vocab_size = size(vars["tok_embeddings.weight"], 1)
    hidden_dim = size(vars["layers.1.feed_forward.w1.weight"], 1)

    shared_weights = !("output.weight" in keys(vars))
    seq_len = get(params_dict, "seq_len", 2048)

    # Construct config header
    p = Config(
    dim,        # transformer dimension
    hidden_dim, # for ffn layers
    n_layers,   # number of layers
    n_heads,    # number of query heads
    n_kv_heads, # number of key/value heads (can be < query heads because of multiquery)
    vocab_size, # vocabulary size, usually 256 (byte-level)
    seq_len,    # max sequence length
    shared_weights
    )

    @info "dim = $dim"
    @info "hidden_dim = $hidden_dim"
    @info "n_layers = $n_layers"
    @info "n_heads = $n_heads"
    @info "n_kv_heads = $n_kv_heads"
    @info "seq_len = $seq_len"
    @info "shared_weights = $shared_weights"
    @info "vocab_size = $vocab_size"

    # Clump tensors together
    wq = T(undef, (dim, dim, n_layers))
    # Materalize transpose view with copy to CPU memory
    wtmp = Array{S}(undef, (dim, dim))
    for l in 0:n_layers-1
        wtmp[:,:] = vars["layers.$l.attention.wq.weight"]'
        wq[:,:,l+1] = wtmp
    end
    wk = T(undef, (dim, dim, n_layers))
    for l in 0:n_layers-1
        wtmp[:,:] = vars["layers.$l.attention.wk.weight"]'
        wk[:,:,l+1] = wtmp
    end
    wv = T(undef, (dim, dim, n_layers))
    for l in 0:n_layers-1
        wtmp[:,:] = vars["layers.$l.attention.wv.weight"]'
        wv[:,:,l+1] = wtmp
    end
    wo = T(undef, (dim, dim, n_layers))
    for l in 0:n_layers-1
        wtmp[:,:] = vars["layers.$l.attention.wo.weight"]'
        wo[:,:,l+1] = wtmp
    end

    wtmp = Array{S}(undef, (dim, hidden_dim))
    w1 = T(undef, (dim, hidden_dim, n_layers))
    for l in 0:n_layers-1
        wtmp[:,:] = vars["layers.$l.feed_forward.w1.weight"]'
        w1[:,:,l+1] = wtmp
    end
    w3 = T(undef, (dim, hidden_dim, n_layers))
    for l in 0:n_layers-1
        wtmp[:,:] = vars["layers.$l.feed_forward.w3.weight"]'
        w3[:,:,l+1] = wtmp
    end
    wtmp = reshape(wtmp, (hidden_dim, dim))
    w2 = T(undef, (hidden_dim, dim, n_layers))
    for l in 0:n_layers-1
        wtmp[:,:] = vars["layers.$l.feed_forward.w2.weight"]'
        w2[:,:,l+1] = wtmp
    end

    rms_att_weight = T(undef, (dim, n_layers))
    for l in 0:n_layers-1
        rms_att_weight[:,l+1] = vars["layers.$l.attention_norm.weight"]
    end
    rms_ffn_weight = T(undef, (dim, n_layers))
    for l in 0:n_layers-1
        rms_ffn_weight[:,l+1] = vars["layers.$l.ffn_norm.weight"]
    end

    f_cos, f_sin = RoPE(dim÷n_heads, seq_len)

    w = TransformerWeights(config=p,
        token_embedding_table = T(copy(vars["tok_embeddings.weight"]')),
        rms_att_weight=rms_att_weight,
        rms_ffn_weight=rms_ffn_weight,
        wq=wq, wk=wk, wv=wv, wo=wo,
        w1=w1, w2=w2, w3=w3,
        # final rmsnorm
        rms_final_weight = T(copy(vars["norm.weight"])),
        freq_cis_real = T(f_cos), freq_cis_imag = T(f_sin),
        wcls = T(copy(vars[shared_weights ? "tok_embeddings.weight" : "output.weight"]'))
    )
    return p, w
end


"current wave of activations"
@kwdef struct RunState{T, Vec<:AbstractVector{T}, Arr3<:AbstractArray{T,3}}
    x::Vec      # activation at current time stamp (dim,)
    xb::Vec     # same, but inside a residual branch (dim,)
    xb2::Vec    # an additional buffer just for convenience (dim,)
    hb::Vec     # buffer for hidden dimension in the ffn (hidden_dim,)
    hb2::Vec    # buffer for hidden dimension in the ffn (hidden_dim,)
    q::Vec      # query (dim,)
    k::Vec      # key (dim,)
    v::Vec      # value (dim,)
    att::Vec    # buffer for scores/attention values (seq_len,)
    logits::Vec # output logits
    # kv cache
    key_cache::Arr3   # (dim, seq_len, layer)
    value_cache::Arr3 # (dim, seq_len, layer)
end

RunState(p::Config) = RunState(Array{Float32}, p)
RunState(::Type{T}, p::Config) where {T<:AbstractArray} = RunState(;
    x = T(undef, (p.dim,)),
    xb = T(undef, (p.dim,)),
    xb2 = T(undef, (p.dim,)),
    hb = T(undef, (p.hidden_dim,)),
    hb2 = T(undef, (p.hidden_dim,)),
    q = T(undef, (p.dim,)),
    k = T(undef, (p.dim,)),
    v = T(undef, (p.dim,)),
    att = T(undef, (p.seq_len,)),
    logits = T(undef, (p.vocab_size,)),
    key_cache = T(undef, (p.dim, p.seq_len, p.n_layers)),
    value_cache = T(undef, (p.dim, p.seq_len, p.n_layers)),
)
