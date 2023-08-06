module MetalExt

import Base: convert
import LanguageModels: RunState, TransformerWeights
using LinearAlgebra
using Metal
using BFloat16s

LinearAlgebra.mul!(c::MtlMatrix{T}, a::Adjoint{T,MtlMatrix{T}}, b::MtlMatrix{T}) where T = Metal.MPS.matmul!(c, a.parent, b, true, true, true, false)
LinearAlgebra.mul!(c::MtlVector{T}, a::Adjoint{T,MtlMatrix{T}}, b::MtlVector{T}) where T = Metal.MPS.matmul!(reshape(c, size(c,1), 1), a.parent, reshape(b, size(b,1), 1), true, true, true, false)

function MtlArray{T}(w::TransformerWeights) where T
    return TransformerWeights{T, MtlArray{T,1}, MtlArray{T,2}, MtlArray{T,3}}(
        config=config,
        token_embedding_table = MtlArray{T}(w.token_embedding_table),
        rms_att_weight = MtlArray{T}(w.rms_att_weight),
        rms_ffn_weight = MtlArray{T}(w.rms_ffn_weight),
        wq = MtlArray{T}(w.wq),
        wk = MtlArray{T}(w.wk),
        wv = MtlArray{T}(w.wv),
        wo = MtlArray{T}(w.wo),
        w1 = MtlArray{T}(w.w1),
        w2 = MtlArray{T}(w.w2),
        w3 = MtlArray{T}(w.w3),
        rms_final_weight =  MtlArray{T}(w.rms_final_weight),
        freq_cis_real =  MtlArray{T}(w.freq_cis_real),
        freq_cis_imag =  MtlArray{T}(w.freq_cis_imag),
        wcls =  MtlArray{T}(w.wcls)
    )
end

function MtlArray{T}(s::RunState) where T
    return RunState{T, MtlArray{T,1}, MtlArray{T, 3}}(
        x = MtlArray{T}(s.x),
        xb = MtlArray{T}(s.xb),
        xb2 = MtlArray{T}(s.xb2),
        hb = MtlArray{T}(s.hb),
        hb2 = MtlArray{T}(s.hb2),
        q = MtlArray{T}(s.q),
        k = MtlArray{T}(s.k),
        v = MtlArray{T}(s.v),
        att = MtlArray{T}(s.att),
        logits = MtlArray{T}(s.logits),
        key_cache = MtlArray{T}(s.key_cache),
        value_cache = MtlArray{T}(s.value_cache)
    )
end

end # module