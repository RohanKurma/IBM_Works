import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import meeting_assistant as m

class MeetingTests(unittest.TestCase):
    def test_transcript_bounds(self):
        for value in (' ', 'x' * 24001):
            with self.assertRaises(ValueError):
                m.validate_transcript(value)
        self.assertEqual(m.validate_transcript(' hello '), 'hello')

    def test_missing_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'WATSONX_APIKEY'):
                m.generate_notes('A meeting')

    def test_transcription_and_empty_speech(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audio.wav'
            path.touch()
            model = Mock()
            model.transcribe.return_value = {'text': ' Hello team '}
            with patch.object(m, 'whisper_model', return_value=model):
                self.assertEqual(m.transcribe(path), 'Hello team')
                model.transcribe.assert_called_once_with(str(path), fp16=False)
                model.transcribe.return_value = {'text': ''}
                with self.assertRaisesRegex(ValueError, 'No speech'):
                    m.transcribe(path)

    def test_missing_audio(self):
        with self.assertRaises(ValueError):
            m.transcribe('/not-a-real-meeting.wav')

    def test_exports_preserve_previous_meetings(self):
        with tempfile.TemporaryDirectory() as directory:
            first = m.export_results('First transcript', '# First notes', directory)
            second = m.export_results('Second transcript', '# Second notes', directory)
            self.assertNotEqual(first, second)
            self.assertEqual(Path(first[0]).read_text(), 'First transcript\n')
            self.assertEqual(Path(first[1]).read_text(), '# First notes\n')
            self.assertEqual(Path(second[0]).read_text(), 'Second transcript\n')
