from django.contrib import admin
from .models import Meeting, Speaker, Transcript, Summary
# Register your models here.

admin.site.register(Meeting)
admin.site.register(Speaker)
admin.site.register(Transcript)
admin.site.register(Summary)
