import json

import numpy as np
from django.db import models


class Individual(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    photo = models.ImageField(upload_to="individual_photos/")
    face_encoding = models.TextField(blank=True, null=True)  # Store the encoding as a JSON string
    duplicate = models.TextField(blank=True, null=True)

    class Meta:
        unique_together = [["first_name", "last_name"]]

    def __str__(self):
        return f"{self.first_name} {self.last_name}"

    def set_face_encoding(self, encoding):
        """Serialize and store the face encoding"""
        if isinstance(encoding, np.ndarray):
            # Convert ndarray to list before serialization
            encoding = encoding.tolist()
        self.face_encoding = json.dumps(encoding)
        self.save()

    def get_face_encoding(self):
        return json.loads(self.face_encoding) if self.face_encoding else None
