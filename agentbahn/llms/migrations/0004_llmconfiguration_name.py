from django.db import migrations
from django.db import models


def populate_llm_configuration_names(apps, schema_editor) -> None:
    llm_configuration = apps.get_model("llms", "LlmConfiguration")
    for config in llm_configuration.objects.all():
        generated_name = f"{config.provider}/{config.llm_name}".strip("/")
        config.name = generated_name or "LLM configuration"
        config.save(update_fields=["name"])


class Migration(migrations.Migration):
    dependencies = [
        ("llms", "0003_llmconfiguration_lm_backend_path"),
    ]

    operations = [
        migrations.AddField(
            model_name="llmconfiguration",
            name="name",
            field=models.CharField(default="LLM configuration", max_length=255),
        ),
        migrations.RunPython(
            populate_llm_configuration_names,
            migrations.RunPython.noop,
        ),
        migrations.AlterField(
            model_name="llmconfiguration",
            name="name",
            field=models.CharField(max_length=255),
        ),
    ]
