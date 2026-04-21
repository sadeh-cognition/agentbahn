from __future__ import annotations

from django.contrib.auth.hashers import identify_hasher
from django.contrib.auth.hashers import make_password
from django.core.exceptions import ValidationError
from django.db import models


def hash_secret(value: str) -> str:
    try:
        identify_hasher(value)
    except ValueError:
        return make_password(value)
    return value


class LlmConfiguration(models.Model):
    provider = models.CharField(max_length=255)
    llm_name = models.CharField(max_length=255)
    api_key_hash = models.CharField(max_length=255)
    date_created = models.DateTimeField(auto_now_add=True)
    date_updated = models.DateTimeField(auto_now=True)

    def clean(self) -> None:
        self.provider = self.provider.strip()
        self.llm_name = self.llm_name.strip()
        self.api_key_hash = self.api_key_hash.strip()
        if not self.provider:
            raise ValidationError({"provider": "This field cannot be blank."})
        if not self.llm_name:
            raise ValidationError({"llm_name": "This field cannot be blank."})
        if not self.api_key_hash:
            raise ValidationError({"api_key_hash": "This field cannot be blank."})

    def save(self, *args: object, **kwargs: object) -> None:
        self.full_clean()
        self.api_key_hash = hash_secret(self.api_key_hash)
        super().save(*args, **kwargs)
