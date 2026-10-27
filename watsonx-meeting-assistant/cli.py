import argparse
from pathlib import Path
from dotenv import load_dotenv
from meeting_assistant import transcribe, generate_notes, export_results


def main():
    p = argparse.ArgumentParser(description='Create meeting transcripts and notes.')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--audio', type=Path)
    source.add_argument('--transcript', type=Path)
    p.add_argument('--model', choices=['tiny', 'base', 'small'], default='base')
    p.add_argument('--transcribe-only', action='store_true')
    p.add_argument('--output-dir', type=Path, default=Path('outputs'))
    args = p.parse_args()
    load_dotenv(Path(__file__).with_name('.env'))
    try:
        text = transcribe(args.audio, args.model) if args.audio else args.transcript.read_text(encoding='utf-8').strip()
        if not text:
            raise ValueError('The transcript is empty.')
        # Preserve the transcript even when IBM is unavailable.
        transcript_path, notes_path = export_results(text, 'Notes have not been generated.', args.output_dir)
        print('Transcript:', transcript_path)
        if not args.transcribe_only:
            notes = generate_notes(text)
            Path(notes_path).write_text(notes + '\n', encoding='utf-8')
            print('Notes:', notes_path)
    except ValueError as exc:
        p.exit(1, f'{exc}\n')
    except Exception:
        p.exit(1, 'Processing failed. Check the input, FFmpeg, model downloads and watsonx configuration. Any transcript already saved remains available.\n')


if __name__ == '__main__':
    main()
