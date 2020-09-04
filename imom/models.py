import os
from django import forms
from django.db import models

# Create your models here.
class Meeting(models.Model):
    name = models.CharField(max_length=50)
    audio = models.FileField(upload_to='audio')
    summary = models.TextField(blank=True, default='')

    def __str__(self):
        return self.name

    def delete(self, *args, **kwargs):
        os.remove(self.audio.path)
        super(Meeting,self).delete(*args,**kwargs)

class Speaker(models.Model):
    name = models.CharField(max_length=20)
    mid = models.ForeignKey(Meeting, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

class Transcript(models.Model):
    text = models.CharField(max_length=255)
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
        fields = '__all__'
    
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