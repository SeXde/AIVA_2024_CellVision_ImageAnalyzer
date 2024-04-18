import unittest
import src.token_validator as token_validator


class TestTokenValidator(unittest.TestCase):
    def setUp(self):
        self.invalid_tokens = ['asd', 'asd1', 'asd3']
        self.token_validator = token_validator.TokenValidator()

    def test_is_invalid_token(self):
        for invalid_token in self.invalid_tokens:
            with self.subTest(msg=f'Checking "{invalid_token}"'):
                self.assertFalse(self.token_validator.validate_token(invalid_token))

    def test_is_valid_token(self):
        for valid_token in self.token_validator.valid_tokens:
            with self.subTest(msg=f'Checking "{valid_token}"'):
                self.assertTrue(self.token_validator.validate_token(valid_token))
