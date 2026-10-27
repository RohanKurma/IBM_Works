# Mistral + gTTS Personal Storyteller

Generate a beginner-friendly educational story from a topic using an IBM watsonx.ai text-generation model, then save an English MP3 narration with Google Text-to-Speech (gTTS).

This project converts an IBM Skills Network learning notebook into a reusable Python command-line application. It includes environment-based credentials, input validation, separate text and audio functions, unique output files, and a cleaned copy of the original lab.

## Setup

Requires Python 3.11, internet access, an IBM Cloud API key, a watsonx.ai project, and access to a supported text-generation model. Model availability depends on your account and region. The original notebook uses Mistral Small; its uploaded filename says Mixtral. This project does not assume that model is still available.

```bash
git clone https://github.com/RohanKurma/IBM_Works.git
cd IBM_Works/mistral-gtts-storyteller
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, `WATSONX_MODEL_ID`, and your regional `WATSONX_URL`. Never commit `.env`.

## Run

```bash
python storyteller.py "the life cycle of butterflies"
python storyteller.py "how trees grow" --text-only
python storyteller.py "the water cycle" --output-dir outputs/water-cycle
```

The application prints the story and writes a uniquely named `.txt` file and, unless `--text-only` is selected, an `.mp3` file. If speech synthesis fails, the saved text remains available. The 200-300 word length is a prompt target, not an enforced limit. Narration is English; changing a gTTS language alone would not translate the story.

Pipeline: topic → prompt → watsonx text generation → saved story → gTTS → MP3.

## Files

| File | Purpose |
| --- | --- |
| `storyteller.py` | Reusable generation/audio functions and CLI |
| `.env.example` | Credential and model configuration template |
| `requirements.txt` | Runtime dependency constraints |
| `notebooks/original_lab.ipynb` | Source lab with outputs removed |
| `tests/test_storyteller.py` | Offline regression tests |
| `NOTICE.md` | Source attribution |

## Original notebook

The preserved notebook is the original Skills Network lab, not the recommended local entry point. Its `skills-network` project and credential-free setup are specific to the hosted lab. Use the CLI above for local execution. Install Jupyter separately to view the notebook. Its original background/model descriptions are preserved as source material, not verified product claims.

## Validation

```bash
python -m unittest discover -s tests -v
```

Tests use fake provider modules to validate SDK parameter wiring, empty responses, credential checks and audio calls without paid requests. Live generation and speech require your configured services and have not been verified in this conversion environment. Dependencies are bounded but not a fully locked environment.

## Service considerations

Topics are sent to IBM watsonx.ai; generated stories are sent to Google's speech service when narration is enabled. IBM inference may incur charges. Generated facts need review before educational use. If generation fails, check your model ID, regional URL, project permissions and API key. For audio failures, check internet access or use `--text-only`.

## Attribution

Based on the IBM Skills Network lab “Use Mistral and gTTS to Create Your Personal Storyteller,” authored by Ricky Shi, with contributors Hailey Quach, Joseph Santarcangelo, and Sadiq Azmi. Original notebook copyright notices are retained. See `NOTICE.md`.
