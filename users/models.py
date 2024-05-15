from django.db import models
import numpy as np
import json

class User(models.Model):
    name = models.CharField(max_length=100)

class Record(models.Model):
    content = models.TextField()
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    embedding = models.JSONField(default=list)  # Store embeddings as JSON

    def set_embedding(self, embedding):
        self.embedding = json.dumps(embedding)

    def get_embedding(self):
        return np.array(json.loads(self.embedding))
