# Read sentencepiece model
# Download protobufs spec from sentencepiece repo

if !isfile("sentencepiece_model.proto")
    Downloads.download("https://raw.githubusercontent.com/google/sentencepiece/635fe8423a249b6e081aacd290d8aef7476c6a28/src/sentencepiece_model.proto",
    "sentencepiece_model.proto")
end

# Generate protobuf interface 
if !isfile("sentencepiece/sentencepiece_model_pb.jl")
    protojl("sentencepiece_model.proto", ".", ".")
end

# Load interface
include("sentencepiece/sentencepiece_model_pb.jl")

function load_sentencepiece_model(filename, vocab_size)
    tokenizer_model = open(filename) do f
        d = ProtoDecoder(f)
        decode(d, ModelProto)
    end
    pieces = [p.piece for p in tokenizer_model.pieces]
    scores = [p.score for p in tokenizer_model.pieces]

    if !(vocab_size == length(pieces) == length(scores))
        @warn "Expected $vocab_size tokens, but found $(length(pieces)) tokens in $filename"
    end

    # Cosmetric rewriting of pieces
    for i in 0x00:0xff
        pieces[i+4] = string(Char(i))
    end
    for (i, p) in enumerate(pieces)
        pieces[i] = replace(p, "▁"=>" ") # Rewrite sentencepiece's ▁ to space
    end

    DigramEncodingTokenizer(pieces, scores)
end
