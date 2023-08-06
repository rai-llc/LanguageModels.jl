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
struct DigramEncodingTokenizer{T, S<:Real}
    alphabet::Vector{T}
    scores::Vector{S}
    output::Vector{T}
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
    len = length(text)
    skip = 0
    for (idx, char) in enumerate(text)
        if skip > 0
            skip -= 1
            continue
        end
        # First look for special tokens
        for tok_id in 1:3
            tok = alphabet[tok_id]
            if idx + length(tok) < len && text[idx:idx+length(tok)-1] == tok
                @info "Matched special token $tok"
                push!(tokens, tok_id)
                skip = length(tok)-1
                @goto next_character
            end
        end
        # Then look for everything else
        id = findfirst(isequal(char), alphabet)
        if !isnothing(id)
            push!(tokens, id)
        else # Fallback; manually encode each code unit
            for cu in codeunits(string(char))
                push!(tokens, cu+4)
            end
        end
        @label next_character
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

# getindex
Base.getindex(t::DigramEncodingTokenizer, i::Int) = t.output[i]
