using ReplMaker

global repl_state=nothing
global repl_tokenizer=nothing
global repl_weights=nothing
global repl_kwargs=Dict()

function repl_driver(prompt)
    global repl_state
    global repl_tokenizer
    global repl_weights
    global repl_kwargs
    _, _, repl_state, repl_tokenizer, repl_weights = main(state=repl_state, tokenizer=repl_tokenizer, weights=repl_weights, prompt=prompt, print_prompt=false; repl_kwargs...)
    return nothing
end

function init_repl(;kwargs...)
    repl_kwargs = Dict(k => v for (k, v) in kwargs)
    initrepl(repl_driver,
        prompt_text="language_model> ",
        prompt_color=:magenta,
        start_key='~',
        mode_name="language_model_mode"
    )
    return nothing
end
