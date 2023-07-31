module LanguageModels

using LinearAlgebra
using LogExpFunctions
using StatsBase

include("artifacts.jl") # Artifacts for managing model files
include("modeldata.jl") # Model data 
include("tokenizer.jl") # Tokenizer types
include("transformer.jl") # Transformer architecture
include("repl.jl")

function rmsnorm!(o, x, weight; ϵ = 1f-5)
    n = length(x)
    ss = 1/√(x⋅x/n + ϵ)
    # normalize and scale
    o .= weight .* (ss .* x)
end

softmax!(x) = LogExpFunctions.softmax!(x,x)

end #module
