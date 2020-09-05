import os
import ast
import io
from django.conf import settings
from subprocess import Popen, PIPE
from django.urls import reverse
from django.views import View
from django.http import FileResponse
from django.shortcuts import render, redirect
from .tasks import prepare_summary, prepare_transcript
from django_celery_results.models import TaskResult
from .models import Meeting, MeetingForm, SpeakerEditForm, Speaker, Transcript, Summary

# Create your views here.
class Home(View):
    meetings = Meeting.objects.all()
    form_class = MeetingForm
    template_name = 'imom/home.html'

    def get(self, request):
        form = self.form_class(auto_id=False)
        for m in self.meetings:
            print(m.transcript_id, m.summary_id)
        return render(request, self.template_name, {"form":form, "meetings":self.meetings, "task_id": "wtf"})
    
    def post(self, request):
        form = self.form_class(request.POST, request.FILES)
        if form.is_valid():
            meeting = form.save(commit=False)
            meeting.save()
            transcript_id = prepare_transcript.delay(meeting.id).task_id
            summary_id = prepare_summary.delay(meeting.id).task_id
            meeting.transcript_id = transcript_id
            meeting.summary_id = summary_id
            meeting.save()
            return redirect(reverse('imom:home'))
        else:
            return render(request, self.template_name, {"form": form, "meetings":self.meetings})

def transcript(request,mid):
    form = SpeakerEditForm(mid=mid)
    meeting = Meeting.objects.get(pk=mid)
    transcripts = Transcript.objects.filter(mid=meeting)
    if request.method == 'POST':
        form = SpeakerEditForm(request.POST)
        if form.is_valid():
            sid = form.cleaned_data['sid'].id
            new_name = form.cleaned_data['new_name']
            speaker = Speaker.objects.get(pk=int(sid))
            speaker.name = new_name
            speaker.save()
            return redirect(reverse('imom:transcript', args=(mid,)))
        
    return render(request, 'imom/transcript.html', {"form":form, "transcripts":transcripts, "meeting":meeting})

def summary(request,mid):
    meeting = Meeting.objects.get(pk=mid)
    summary = Summary.objects.get(mid=meeting)
        
    return render(request, 'imom/summary.html', {"summary":summary})

def download_transcript(request, mid):
    f = io.BytesIO()
    meeting = Meeting.objects.get(pk=mid)
    transcripts = Transcript.objects.filter(mid=meeting)
    for transcript in transcripts:
        t = "%s %s: %s" % (transcript.timestamp_in_hms(),transcript.sid.name, transcript.text)
        f.write(t.encode('utf-8') + b'\n')
    f.seek(0)
    return FileResponse(f, as_attachment=True, filename="%s_transcript.txt" % meeting.name)

def download_summary(request, id):
    f = io.BytesIO()
    summary = Summary.objects.get(pk=id)
    if request.GET['type'] == 'abs':
        f.write(summary.abs_summary.encode('utf-8'))
    else:
        f.write(summary.ext_summary.encode('utf-8'))
    f.seek(0)
    return FileResponse(f, as_attachment=True, filename="%s_summary.txt" % summary.mid.name)

def delete(request,mid):
    meeting = Meeting.objects.get(pk=mid)
    meeting.delete()
    return redirect(reverse('imom:home'))