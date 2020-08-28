from django.contrib import admin
from .models import Meeting, Speaker, Transcript
# Register your models here.

admin.site.register(Meeting)
admin.site.register(Speaker)
admin.site.register(Transcript)