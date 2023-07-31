using Pkg.Artifacts
using Downloads

artifact_toml = joinpath(@__DIR__, "..", "Artifacts.toml")

stories15M_model = artifact_hash("stories15M_model", artifact_toml)

if stories15M_model == nothing || !artifact_exists(stories15M_model)
    stories15M_model = create_artifact() do artifact_dir
        Downloads.download("https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
           joinpath(artifact_dir, "stories15M.bin"))
        Downloads.download("https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin",
           joinpath(artifact_dir, "tokenizer.bin"))
    end

    bind_artifact!(artifact_toml, "stories15M", stories15M_model, lazy=true, force=true)
end

model_path = artifact_path(stories15M_model)
