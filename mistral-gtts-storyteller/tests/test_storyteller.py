import os
import sys
import types
import unittest
from unittest.mock import Mock, patch
import storyteller

class StoryTests(unittest.TestCase):
    def test_blank_topic(self):
        with self.assertRaises(ValueError):
            storyteller.generate_story('   ')

    def test_missing_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'WATSONX_APIKEY'):
                storyteller.generate_story('trees')

    def test_provider_wiring_and_empty_response(self):
        sdk = types.ModuleType('ibm_watsonx_ai')
        sdk.Credentials = Mock()
        foundation = types.ModuleType('ibm_watsonx_ai.foundation_models')
        foundation.ModelInference = Mock()
        model = foundation.ModelInference.return_value
        model.generate_text.return_value = ' A tree story. '
        modules = {'ibm_watsonx_ai': sdk, 'ibm_watsonx_ai.foundation_models': foundation}
        env = {'WATSONX_APIKEY': 'test-only', 'WATSONX_PROJECT_ID': 'test-project', 'WATSONX_MODEL_ID': 'test-model'}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, env, clear=True):
            self.assertEqual(storyteller.generate_story('trees'), 'A tree story.')
            self.assertEqual(foundation.ModelInference.call_args.kwargs['model_id'], 'test-model')
            self.assertIn('trees', model.generate_text.call_args.kwargs['prompt'])
            sdk.Credentials.assert_called_once_with(url='https://us-south.ml.cloud.ibm.com', api_key='test-only')
            model.generate_text.return_value = ' '
            with self.assertRaises(RuntimeError):
                storyteller.generate_story('trees')

    def test_audio_provider(self):
        module = types.ModuleType('gtts')
        module.gTTS = Mock()
        with patch.dict(sys.modules, {'gtts': module}):
            storyteller.save_audio('A story', 'example.mp3')
        module.gTTS.assert_called_once_with(text='A story', lang='en', timeout=30)
        module.gTTS.return_value.save.assert_called_once_with('example.mp3')
