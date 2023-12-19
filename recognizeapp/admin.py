from django.contrib import admin
from .models import Individual


class IndividualAdmin(admin.ModelAdmin):
    list_display = ("first_name", "last_name", "id", "duplicate")
    exclude = ("face_encoding",)
    readonly_fields = ("duplicate",)


admin.site.register(Individual, IndividualAdmin)
