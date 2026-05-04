from django.db import migrations
from django.db import models


class Migration(migrations.Migration):
    dependencies = [
        ("llms", "0002_encrypt_api_key"),
    ]

    operations = [
        migrations.AddField(
            model_name="llmconfiguration",
            name="lm_backend_path",
            field=models.CharField(default="default", max_length=255),
        ),
    ]
