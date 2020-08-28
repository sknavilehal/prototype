import os
import ast
from django.conf import settings
from subprocess import Popen, PIPE
from django.urls import reverse
from django.views import View
#from summarization import summarize_pipline
from django.shortcuts import render, redirect
from .tasks import prepare_transcript, prepare_summary
from .models import Meeting, MeetingForm, SpeakerEditForm, Speaker, Transcript

# Create your views here.
class Home(View):
    form_class = MeetingForm
    template_name = 'imom/home.html'

    def get(self, request):
        meetings = Meeting.objects.all()
        form = self.form_class(auto_id=False)
        return render(request, self.template_name, {"form":form, "meetings":meetings})
    
    def post(self, request):
        form = self.form_class(request.POST, request.FILES)
        if form.is_valid():
            meeting = form.save(commit=False)
            meeting.save()
            prepare_transcript.delay(meeting.id)
            #update_meeting(meeting.id)
            prepare_summary.run(meeting.id)
            return redirect(reverse('imom:home'))
        else:
            return render(request, self.template_name, {"form": form})

def transcript(request,mid):
    form = SpeakerEditForm(mid=mid)
    transcripts = Transcript.objects.filter(mid=Meeting.objects.get(pk=mid))
    if request.method == 'POST':
        form = SpeakerEditForm(request.POST)
        if form.is_valid():
            sid = form.cleaned_data['sid'].id
            new_name = form.cleaned_data['new_name']
            speaker = Speaker.objects.get(pk=int(sid))
            speaker.name = new_name
            speaker.save()
            return redirect(reverse('imom:transcript', args=(mid,)))

    return render(request, 'imom/transcript.html', {"form":form, "transcripts":transcripts})

def delete(request,mid):
    meeting = Meeting.objects.get(pk=mid)
    meeting.delete()
    return redirect(reverse('imom:home'))