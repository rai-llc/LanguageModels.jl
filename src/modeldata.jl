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
