import math
import os
import warnings

import click


def _validate_speed(ctx, param, value):
    # Reject 0, negatives, nan, and inf here, before any NPU model is created.
    if not math.isfinite(value) or value <= 0:
        raise click.BadParameter("must be a finite number greater than 0")
    return value


@click.command
@click.argument('text')
@click.argument('output_path')
@click.option("--file", '-f', is_flag=True, show_default=True, default=False, help="Text is a file")
@click.option('--language', '-l', default='EN_NEWEST', help='Language, defaults to English', type=click.Choice(['EN_NEWEST', 'KR'], case_sensitive=False))
@click.option('--speaker', '-spk', default=None, help='Speaker ID, only for English, leave empty for default, ignored if not English. If English, defaults to "EN-Newest"', type=click.Choice(['EN-Newest']))
@click.option('--speed', '-s', default=1.0, help='Speed (finite, greater than 0), defaults to 1.0', type=float, callback=_validate_speed)
@click.option('--device', '-d', default='auto', help='Device, defaults to auto')
@click.option('--local-files-only', '-c', is_flag=True, show_default=True, default=False, help='local_files_only option for TTS model')
def main(text, file, output_path, language, speaker, speed, device, local_files_only):
    if file:
        if not os.path.exists(text):
            raise FileNotFoundError(f'Trying to load text from file due to --file/-f flag, but file not found. Remove the --file/-f flag to pass a string.')
        else:
            with open(text) as f:
                text = f.read().strip()
    if text == '':
        raise ValueError('You entered empty text or the file you passed was empty.')
    language = language.upper()
    if language == '': language = 'EN_NEWEST'
    if speaker == '': speaker = None
    if (not language == 'EN_NEWEST') and speaker:
        warnings.warn('You specified a speaker but the language is not English.')
    from melotts_mblt.api import TTS
    model = TTS(language=language, device=device, trust_remote_code=True, local_files_only=local_files_only)
    try:
        speaker_ids = model.hps.data.spk2id
        if language == 'EN_NEWEST':
            if not speaker: speaker = 'EN-Newest'
            spkr = speaker_ids[speaker]
        else:
            spkr = speaker_ids[list(speaker_ids.keys())[0]]
        model.tts_to_file(text, spkr, output_path, speed=speed)
    finally:
        # run_tts() is also called in-process (e.g. the Model Zoo bridge), so release the NPU backends explicitly.
        model.dispose()
