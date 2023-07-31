using ReplMaker

function talkto(;kwargs...)
    initrepl(prompt->main(prompt=prompt; kwargs...),
        prompt_text="language_model> ",
        prompt_color=:magenta,
        start_key='~',
        mode_name="language_model_mode"
    )
end
