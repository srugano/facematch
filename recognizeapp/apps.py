from django.apps import AppConfig


class RecognizeappConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "recognizeapp"

    def ready(self):
        import recognizeapp.signals  # Import the signals module
