"""Run the local Gradio meeting assistant."""
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
from meeting_assistant import transcribe, generate_notes, export_results

load_dotenv(Path(__file__).with_name('.env'))


def transcribe_ui(audio, size):
    if not audio:
        raise gr.Error('Upload or record audio first.')
    try:
        return transcribe(audio, size), '', None, None
    except ValueError as exc:
        raise gr.Error(str(exc)) from None
    except Exception:
        raise gr.Error('Transcription failed. Check FFmpeg, audio format, model download access and available memory.') from None


def notes_ui(text):
    try:
        notes = generate_notes(text)
        # One private directory per result avoids overwriting other sessions.
        folder = tempfile.mkdtemp(prefix='meeting-assistant-')
        transcript_path, notes_path = export_results(text, notes, folder)
        return notes, transcript_path, notes_path
    except ValueError as exc:
        raise gr.Error(str(exc)) from None
    except Exception:
        raise gr.Error('Notes generation failed. Check watsonx configuration, model access and context limits.') from None


def build_app():
    with gr.Blocks(title='AI Meeting Assistant', delete_cache=(3600, 3600)) as demo:
        gr.Markdown('# AI Meeting Assistant\nTranscribe a recording, review the text, then generate meeting notes.')
        gr.Markdown('Audio is processed locally by Whisper. Clicking Generate notes sends the transcript to IBM watsonx.ai. Use recordings you have permission to process.')
        audio = gr.Audio(sources=['upload', 'microphone'], type='filepath', label='Meeting audio')
        size = gr.Dropdown(['tiny', 'base', 'small'], value='base', label='Whisper model')
        transcription = gr.Button('Transcribe audio')
        transcript = gr.Textbox(label='Review or paste transcript', lines=12)
        summarize = gr.Button('Generate notes', variant='primary')
        notes = gr.Markdown()
        transcript_file = gr.File(label='Download transcript')
        notes_file = gr.File(label='Download notes')
        transcription.click(transcribe_ui, [audio, size], [transcript, notes, transcript_file, notes_file])
        summarize.click(notes_ui, transcript, [notes, transcript_file, notes_file])
        transcript.input(lambda: ('', None, None), outputs=[notes, transcript_file, notes_file])
    return demo


if __name__ == '__main__':
    build_app().queue(default_concurrency_limit=1).launch(server_name='127.0.0.1', server_port=7860, share=False)
