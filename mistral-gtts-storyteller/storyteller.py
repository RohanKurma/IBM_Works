"""Generate educational stories with watsonx.ai and narrate them with gTTS."""
import argparse
import os
from pathlib import Path


def build_prompt(topic):
    topic = topic.strip()
    if not topic:
        raise ValueError('Topic must not be empty.')
    if len(topic) > 1000:
        raise ValueError('Keep the topic under 1,001 characters.')
    return (f'Write an engaging educational story about {topic} for beginners. '
            'Use simple, clear language and interesting facts. Keep it friendly '
            'and encouraging, around 200-300 words. End with a brief summary '
            'of what we learned. Write in English. Return only the story.')


def generate_story(topic):
    prompt = build_prompt(topic)
    required = ('WATSONX_APIKEY', 'WATSONX_PROJECT_ID', 'WATSONX_MODEL_ID')
    missing = [key for key in required if not os.getenv(key, '').strip()]
    if missing:
        raise ValueError('Set these environment variables: ' + ', '.join(missing))
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    model = ModelInference(
        model_id=os.environ['WATSONX_MODEL_ID'],
        credentials=Credentials(
            url=os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com'),
            api_key=os.environ['WATSONX_APIKEY']),
        project_id=os.environ['WATSONX_PROJECT_ID'],
        params={'decoding_method': 'greedy', 'max_new_tokens': 1000})
    story = model.generate_text(prompt=prompt)
    if not isinstance(story, str) or not story.strip():
        raise RuntimeError('The model returned an empty story. Try another topic or model.')
    return story.strip()


def save_audio(story, path):
    from gtts import gTTS
    gTTS(text=story, lang='en', timeout=30).save(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('topic', help='Topic for an English educational story')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--text-only', action='store_true', help='Skip speech synthesis')
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name('.env'))
    try:
        story = generate_story(args.topic)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Unique filenames preserve previous stories.
        from uuid import uuid4
        stem = 'story-' + uuid4().hex[:12]
        text_path = args.output_dir / (stem + '.txt')
        text_path.write_text(story + '\n', encoding='utf-8')
        print(story)
        print(f'\nText saved: {text_path}')
        if not args.text_only:
            try:
                audio_path = args.output_dir / (stem + '.mp3')
                save_audio(story, audio_path)
                print(f'Audio saved: {audio_path}')
            except Exception:
                parser.exit(1, 'Speech synthesis failed. Your text is saved; check connectivity and retry.\n')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    except Exception:
        parser.exit(1, 'Story generation failed. Check your watsonx credentials, region, model access and connection.\n')


if __name__ == '__main__':
    main()
