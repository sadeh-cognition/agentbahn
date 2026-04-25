from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models


ENCRYPTED_API_KEY_PREFIX = "fernet:"


def _api_key_fernet() -> Fernet:
    key_material = settings.LLM_API_KEY_ENCRYPTION_KEY.encode("utf-8")
    fernet_key = base64.urlsafe_b64encode(hashlib.sha256(key_material).digest())
    return Fernet(fernet_key)


def encrypt_api_key(value: str) -> str:
    encrypted_value = _api_key_fernet().encrypt(value.encode("utf-8")).decode("utf-8")
    return f"{ENCRYPTED_API_KEY_PREFIX}{encrypted_value}"


def decrypt_api_key(value: str) -> str:
    if not value.startswith(ENCRYPTED_API_KEY_PREFIX):
        raise ValidationError("LLM API key is not encrypted.")
    encrypted_value = value.removeprefix(ENCRYPTED_API_KEY_PREFIX)
    return _api_key_fernet().decrypt(encrypted_value.encode("utf-8")).decode("utf-8")


class LlmConfiguration(models.Model):
    provider = models.CharField(max_length=255)
    llm_name = models.CharField(max_length=255)
    encrypted_api_key = models.TextField()
    date_created = models.DateTimeField(auto_now_add=True)
    date_updated = models.DateTimeField(auto_now=True)

    def clean(self) -> None:
        self.provider = self.provider.strip()
        self.llm_name = self.llm_name.strip()
        self.encrypted_api_key = self.encrypted_api_key.strip()
        if not self.provider:
            raise ValidationError({"provider": "This field cannot be blank."})
        if not self.llm_name:
            raise ValidationError({"llm_name": "This field cannot be blank."})
        if not self.encrypted_api_key:
            raise ValidationError({"encrypted_api_key": "This field cannot be blank."})

    def save(self, *args: object, **kwargs: object) -> None:
        self.full_clean()
        if not self.encrypted_api_key.startswith(ENCRYPTED_API_KEY_PREFIX):
            self.encrypted_api_key = encrypt_api_key(self.encrypted_api_key)
        super().save(*args, **kwargs)
