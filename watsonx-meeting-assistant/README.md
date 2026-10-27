# AI Meeting Assistant

Turn meeting recordings into editable transcripts and structured Markdown minutes using local OpenAI Whisper, IBM watsonx.ai, LangChain prompt chains, and Gradio.

Based on the project outline in the uploaded Skills Network document, **Build an AI Meeting Assistant: Instant Notes, Zero Worries**. The readable source describes goals and lab instructions rather than implementation code. This repository provides a new implementation of that outline.

## Features

- Upload audio or record from the microphone in a local browser interface.
- Transcribe with Whisper tiny, base, or small; paste existing text to skip audio processing.
- Review the raw transcript before sending it to watsonx.
- Produce summaries, discussion points, decisions, action items with owners/deadlines, and open questions.
- Download transcript text and Markdown notes, or use a command-line workflow.
- Keep API credentials in environment variables and avoid overwriting previous exports.

This is batch processing after recording, not live streaming. It does not identify speakers, translate meetings, or automatically correct transcripts. Review names, dates, decisions and tasks before sharing. Missing owners and deadlines are requested as “Not specified”; LLM output is not guaranteed to follow instructions.

## Setup

Use Python 3.11 and install **FFmpeg**, ensuring `ffmpeg -version` works in your terminal. Whisper uses PyTorch; CPU works, while compatible GPU installations can accelerate transcription. First use downloads model weights and requires internet access and disk space.

```bash
git clone https://github.com/RohanKurma/IBM_Works.git
cd IBM_Works/watsonx-meeting-assistant
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS/Linux.

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`. Set your IBM API key, project ID, regional URL and a text-generation model ID available in your project. Hosted Skills Network credentials do not work as local credentials. watsonx inference may incur charges.

## Browser app

```bash
python app.py
```

Open http://127.0.0.1:7860. Upload/record audio → Transcribe audio → review transcript → Generate notes → download results. The server binds to localhost and does not request a public Gradio share link. Keep this local demo local; authentication and production hosting are not included.

Whisper decodes formats supported by your FFmpeg installation, including common WAV, MP3 and M4A recordings. Try a short clear recording first. CPU processing can take time.

## Command line

```bash
python cli.py --audio meeting.wav
python cli.py --audio meeting.mp3 --model tiny --transcribe-only
python cli.py --transcript transcript.txt --output-dir outputs/team-meeting
```

Exports use unique names in `outputs/`. The CLI saves the transcript before requesting notes, preserving it if summarization fails. In transcription-only mode the companion notes file contains a placeholder. Transcription needs no IBM credentials; notes generation does.

## Architecture and files

Audio → local Whisper → editable transcript → LangChain PromptTemplate + RunnableLambda → watsonx ModelInference → Markdown notes and exports.

| File | Purpose |
| --- | --- |
| `app.py` | Gradio user interface |
| `cli.py` | Batch command line |
| `meeting_assistant.py` | Transcription, validation, notes chain and export functions |
| `.env.example` | Configuration template |
| `requirements.txt` | Dependency constraints |
| `tests/test_meeting_assistant.py` | Offline regression tests |
| `NOTICE.md` | Source and adaptation notes |

## Limits and privacy

Audio stays in the local Whisper workflow; transcripts are sent to IBM only when notes generation is requested. Gradio stores uploaded files and generated downloads in local temporary storage. Generated export directories persist until manually removed; remove sensitive temporary files and `outputs/` when no longer needed. No retention service is implemented.

Input to the notes chain is limited to 24,000 characters and rejected rather than silently truncated. Models have differing context windows; reduce the transcript further if your model rejects it. Long meetings need manual splitting. The model output budget can also truncate long notes. This application does not claim completeness or measured transcription accuracy.

## Validation and troubleshooting

```bash
python -m unittest discover -s tests -v
```

Offline tests cover invalid/oversized inputs, Whisper wiring, empty speech, missing credentials and export preservation. They do not validate live IBM responses or audio recognition accuracy. End-to-end service calls and browser execution require a configured local environment and have not been verified here. Dependency ranges are not a fully locked installation.

If audio fails, verify FFmpeg, model downloads, audio quality and available memory. If notes fail, check `.env`, region, project permissions, model availability and context limits. Use `--transcribe-only` to isolate the transcription step.
