module LanguageModels
using Downloads # Todo: package as Artifact
using JSON
using Pickle
using LinearAlgebra
using LogExpFunctions
using StatsBase
using ProtoBuf
using Mmap # stdlib to memory map weights
include("artifacts.jl") # Artifacts for managing model files
include("modeldata.jl") # nanogpt model data

# Tokenizers
include("sentencepiece.jl") # sentencepiece tokenizer
include("tokenizer.jl") # nanogpt tokenizer

include("transformer.jl") # Transformer architecture
include("repl.jl")

function rmsnorm!(o, x, weight; ϵ = 1f-5)
    n = length(x)
    ss = 1/√(x⋅x/n + ϵ)
    # normalize and scale
    o .= weight .* (ss .* x)
end

softmax!(x) = LogExpFunctions.softmax!(x,x)

# Automatically initialize custom REPL when module is initialized
function __init__()
    if isdefined(Base, :active_repl)
        init_repl()
    end
end

end #module
