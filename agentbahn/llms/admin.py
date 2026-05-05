from __future__ import annotations

from django.contrib import admin

from agentbahn.llms.models import LlmConfiguration


@admin.register(LlmConfiguration)
class LlmConfigurationAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "provider",
        "llm_name",
        "lm_backend_path",
        "date_updated",
    )
    list_filter = ("provider", "lm_backend_path")
    search_fields = ("name", "provider", "llm_name", "lm_backend_path")
    readonly_fields = ("date_created", "date_updated")
    fieldsets = (
        (
            None,
            {
                "fields": (
                    "name",
                    "provider",
                    "llm_name",
                    "lm_backend_path",
                    "encrypted_api_key",
                )
            },
        ),
        (
            "Timestamps",
            {
                "fields": ("date_created", "date_updated"),
            },
        ),
    )
