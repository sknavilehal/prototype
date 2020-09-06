import os
import io
from django import forms
from django.db import models
from django.core import files
from django_celery_results.models import TaskResult
from pydub import AudioSegment
# Create your models here.
class Meeting(models.Model):
    name = models.CharField(max_length=50)
    audio = models.FileField(upload_to='audio')
    date = models.DateField()
    transcript_id = models.CharField(null=True, max_length=100)
    summary_id = models.CharField(null=True, max_length=100)

    def __str__(self):
        return self.name

    def delete(self, *args, **kwargs):
        os.remove(self.audio.path)
        super(Meeting,self).delete(*args,**kwargs)
    
    def save(self, *args, **kwargs):
        filename = self.audio.name
        if filename.endswith('.mp3'):
            src = files.File(self.audio.open())
            dst = io.BytesIO()
            sound = AudioSegment.from_mp3(src); src.close()
            sound.export(dst, format="wav")
            #meeting.audio.save(filename[:-4]+'.wav', dst, save=False)
            self.audio = files.File(dst, name=filename[:-4]+'.wav')
        super().save(*args, **kwargs)

class Speaker(models.Model):
    name = models.CharField(max_length=20)
    mid = models.ForeignKey(Meeting, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

class Summary(models.Model):
    abs_summary = models.TextField(blank=True, default='')
    ext_summary = models.TextField(blank=True, default='')
    mid = models.ForeignKey(Meeting, on_delete=models.CASCADE)

class Transcript(models.Model):
    text = models.TextField(blank=True, default='')
    timestamp = models.IntegerField()
    sid = models.ForeignKey(Speaker, on_delete=models.CASCADE)
    mid = models.ForeignKey(Meeting, on_delete=models.CASCADE)

    class Meta:
        ordering = ['timestamp']
    
    def timestamp_in_hms(self):
        millis = self.timestamp
        seconds=(millis/1000)%60
        seconds = int(seconds) 
        minutes=(millis/(1000*60))%60
        minutes = int(minutes)
        hours=(millis/(1000*60*60))%24

        return "[%d:%02d:%02d] " % (hours, minutes, seconds)

    def __str__(self):
        ts = self.timestamp_in_hms()

        return "<strong>%s %s:</strong> %s" % (ts, self.sid.name, self.text) 

class MeetingForm(forms.ModelForm):
    class Meta:
        model = Meeting
        fields = {'name', 'audio', 'date'}
        widgets = {'date': forms.DateInput(attrs={'class': 'form-control mb-2 mr-sm-2 datepicker'})}

    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['name'].widget.attrs.update({'class': 'form-control mb-2 mr-sm-2'})
        self.fields['audio'].widget.attrs.update({'class': 'mb-2 mr-sm-2'})

class SpeakerEditForm(forms.Form):
    def __init__(self, *args, **kwargs):
        mid = kwargs.pop('mid', None)
        super().__init__(*args, **kwargs)
        if mid: self.fields['sid'].queryset = Speaker.objects.filter(mid=mid)
        self.fields['sid'].widget.attrs.update({'class': 'form-control mb-2 mr-sm-2'})
        self.fields['new_name'].widget.attrs.update({'class': 'mb-2 mr-sm-2'})
    
    sid = forms.ModelChoiceField(queryset=Speaker.objects.all())
    new_name = forms.CharField(max_length=20)