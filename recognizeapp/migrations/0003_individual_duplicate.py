# Generated by Django 3.2.23 on 2023-12-19 04:37

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("recognizeapp", "0002_individual_face_encoding"),
    ]

    operations = [
        migrations.AddField(
            model_name="individual",
            name="duplicate",
            field=models.TextField(blank=True, null=True),
        ),
    ]
