from django.db import models


class Individual(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    photo = models.ImageField(upload_to="individual_photos/")

    class Meta:
        unique_together = [["first_name", "last_name"]]

    def __str__(self):
        return f"{self.first_name} {self.last_name}"
