using Pkg.Artifacts
using Downloads
using ProgressMeter

artifact_toml = joinpath(@__DIR__, "..", "Artifacts.toml")

function downloader(url, dir, filename; p=ProgressMeter.Progress(0, ""))
    p = ProgressMeter.Progress(0, "Downloading "*filename)
    Downloads.download(url, joinpath(dir, filename), progress=(n,x)->(p.n=n; if x>0 update!(p, x) end))
end



# Tinyllamas

stories15M_model = artifact_hash("stories15M_model", artifact_toml)
if isnothing(stories15M_model) || !artifact_exists(stories15M_model)
    stories15M_model = create_artifact() do artifact_dir
        downloader("https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
           artifact_dir, "stories15M.bin", )
        downloader("https://github.com/karpathy/llama2.c/raw/master/tokenizer.model",
           artifact_dir, "tokenizer.model", )
    end
end
if isnothing(stories15M_model)
    bind_artifact!(artifact_toml, "stories15M_model", stories15M_model, lazy=true)
end

stories42M_model = artifact_hash("stories42M_model", artifact_toml)
if isnothing(stories42M_model) || !artifact_exists(stories42M_model)
    stories42M_model = create_artifact() do artifact_dir
        downloader("https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin",
           artifact_dir, "stories42M.bin", )
        downloader("https://github.com/karpathy/llama2.c/raw/master/tokenizer.model",
           artifact_dir, "tokenizer.model", )
    end
end
if isnothing(stories42M_model)
    bind_artifact!(artifact_toml, "stories42M_model", stories42M_model, lazy=true)
end

stories110M_model = artifact_hash("stories110M_model", artifact_toml)
if isnothing(stories110M_model) || !artifact_exists(stories110M_model)
    stories110M_model = create_artifact() do artifact_dir
        downloader("https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin",
           artifact_dir, "stories110M.bin", )
        downloader("https://github.com/karpathy/llama2.c/raw/master/tokenizer.model",
           artifact_dir, "tokenizer.model", )
    end
end
if isnothing(stories110M_model)
    bind_artifact!(artifact_toml, "stories110M_model", stories110M_model, lazy=true)
end

