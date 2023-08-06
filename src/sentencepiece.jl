# Read sentencepiece model
# Download protobufs spec from sentencepiece repo
let
    if !isfile(joinpath(@__DIR__, "sentencepiece/sentencepiece_model_pb.jl"))
        proto_file = joinpath(@__DIR__, "sentencepiece_model.proto")
        if !isfile(proto_file)
            using Downloads
            Downloads.download("https://raw.githubusercontent.com/google/sentencepiece/635fe8423a249b6e081aacd290d8aef7476c6a28/src/sentencepiece_model.proto", proto_file)
        end
        protojl(proto_file, ".", ".")
    end
end

# Load interface
include("sentencepiece/sentencepiece_model_pb.jl")

function load_sentencepiece_tokenizer(filename)
    tokenizer_model = open(filename) do f
        d = ProtoDecoder(f)
        decode(d, ModelProto)
    end
    pieces = [p.piece for p in tokenizer_model.pieces]
    scores = [p.score for p in tokenizer_model.pieces]

    for (i, p) in enumerate(pieces)
        pieces[i] = replace(p, "▁"=>" ") # Rewrite sentencepiece's ▁ to space
    end
    for i in 0x00:0xff
        pieces[i+4] = string(Char(i))
    end
    
    output = copy(pieces)

    # Cosmetic rewriting of pieces
    output[1] = "�" #  Unknown token -> Unknown character character
    output[2] = string(Char(0x98)) # Beginning of sentence -> Start of string character
    output[3] = string(Char(0x9c)) # End of sentence -> String terminator character
 
    DigramEncodingTokenizer(pieces, scores, output)
end
