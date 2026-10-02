# WebUI by mrfakename <X @realmrfakename / HF @mrfakename>
# Demo also available on HF Spaces: https://huggingface.co/spaces/mrfakename/MeloTTS
import io

import gradio as gr

# os.system('python -m unidic download')
print("Make sure you've downloaded unidic (python -m unidic download) for this WebUI to work.")
from melotts_mblt.api import TTS

speed = 1.0
import click

device = 'auto'
LANGUAGES = ('EN_NEWEST', 'KR')


def _load_models():
    """Load every WebUI model, releasing the ones already created if a later one fails."""
    loaded = {}
    try:
        for language in LANGUAGES:
            loaded[language] = TTS(language=language, device=device, trust_remote_code=True)
    except BaseException:
        _dispose_models(loaded)
        raise
    return loaded


def _dispose_models(loaded):
    """Release the NPU backends of ``loaded`` and empty it."""
    while loaded:
        _, model = loaded.popitem()
        model.dispose()


models = _load_models()


def _build_demo():
    """Build the Gradio UI. Runs after the models exist, so callers must release them if this fails."""
    global speaker_ids, default_text_dict
    speaker_ids = models['EN_NEWEST'].hps.data.spk2id

    default_text_dict = {
        'EN_NEWEST': 'The field of text-to-speech has seen rapid development recently.',
        'KR': '최근 텍스트 음성 변환 분야가 급속도로 발전하고 있습니다.',
    }

    def synthesize(speaker, text, speed, language, progress=gr.Progress()):
        bio = io.BytesIO()
        models[language].tts_to_file(text, models[language].hps.data.spk2id[speaker], bio, speed=speed, pbar=progress.tqdm, format='wav')
        return bio.getvalue()
    def load_speakers(language, text):
        if text in list(default_text_dict.values()):
            newtext = default_text_dict[language]
        else:
            newtext = text
        return gr.update(value=list(models[language].hps.data.spk2id.keys())[0], choices=list(models[language].hps.data.spk2id.keys())), newtext
    with gr.Blocks() as demo:
        gr.Markdown('# MeloTTS WebUI\n\nA WebUI for MeloTTS.')
        with gr.Group():
            speaker = gr.Dropdown(speaker_ids.keys(), interactive=True, value='EN-Newest', label='Speaker')
            language = gr.Radio(['EN_NEWEST', 'KR'], label='Language', value='EN_NEWEST')
            speed = gr.Slider(label='Speed', minimum=0.1, maximum=10.0, value=1.0, interactive=True, step=0.1)
            text = gr.Textbox(label="Text to speak", value=default_text_dict['EN_NEWEST'])
            language.input(load_speakers, inputs=[language, text], outputs=[speaker, text])
        btn = gr.Button('Synthesize', variant='primary')
        aud = gr.Audio(interactive=False)
        btn.click(synthesize, inputs=[speaker, text, speed, language], outputs=[aud])
        gr.Markdown('WebUI by [mrfakename](https://twitter.com/realmrfakename).')
    return demo


try:
    demo = _build_demo()
except BaseException:
    # Import is aborting after the NPU-backed models were loaded; release them before re-raising.
    _dispose_models(models)
    raise


@click.command()
@click.option('--share', '-s', is_flag=True, show_default=True, default=False, help="Expose a publicly-accessible shared Gradio link usable by anyone with the link. Only share the link with people you trust.")
@click.option('--host', '-h', default=None)
@click.option('--port', '-p', type=int, default=None)
def main(share, host, port):
    # run_ui() can launch the WebUI in-process more than once; reload the models released by a previous run.
    if not models:
        models.update(_load_models())
    try:
        demo.queue(api_open=False).launch(share=share, server_name=host, server_port=port)
    finally:
        # Serving ended (or failed): release every model's NPU backends instead of leaving them to process exit.
        _dispose_models(models)

if __name__ == "__main__":
    main()
