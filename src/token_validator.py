class TokenValidator:
    def __init__(self):
        self.valid_tokens = {'dev-token-1738-r3mi-b0i2z'}

    def validate_token(self, token: str) -> bool:
        return token in self.valid_tokens
