from django.db import migrations
from django.db import models


def delete_existing_llm_configuration(apps, schema_editor) -> None:
    del schema_editor
    llm_configuration = apps.get_model("llms", "LlmConfiguration")
    llm_configuration.objects.all().delete()


class Migration(migrations.Migration):
    dependencies = [
        ("llms", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(
            delete_existing_llm_configuration,
            reverse_code=migrations.RunPython.noop,
        ),
        migrations.RenameField(
            model_name="llmconfiguration",
            old_name="api_key_hash",
            new_name="encrypted_api_key",
        ),
        migrations.AlterField(
            model_name="llmconfiguration",
            name="encrypted_api_key",
            field=models.TextField(),
        ),
    ]
