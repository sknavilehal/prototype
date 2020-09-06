from __future__ import absolute_import, unicode_literals

import ast
import os
import traceback
from celery import shared_task
from django.conf import settings
from subprocess import Popen, PIPE
from .models import Speaker, Meeting, Transcript, Summary
from django_celery_results.models import TaskResult
from celery_progress.backend import ProgressRecorder

@shared_task(bind=True)
def prepare_transcript(self, mid, task_id=None):
    progress_recorder = ProgressRecorder(self)
    progress_recorder.set_progress(1,2)
    meeting = Meeting.objects.get(pk=mid)
    pipeline_path = os.path.join(settings.BASE_DIR, 'transcript','pipeline.py')
    process = Popen(['python', pipeline_path, '--filename', meeting.audio.path], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    stdout, stderr = stdout.decode('utf-8').strip(), stderr.decode('utf-8').strip()
    print(stdout)
    print(stderr)
    try:
        output = ast.literal_eval(stdout)
    except:
        traceback.print_exc()
        output = {0 : [(0,"There was an error while generating the transcript")]}
    for speaker in output.keys():
        s = Speaker(name='speaker %d' % speaker, mid=meeting)
        s.save()
        for t in output[speaker]:
            entry = Transcript(timestamp=t[0],text=t[1],sid=s,mid=meeting)
            entry.save()

    wavs = [os.path.join(settings.BASE_DIR,w) for w in os.listdir() if w.endswith('.wav')]
    for wav in wavs:
        os.remove(wav)

    progress_recorder.set_progress(2,2)
    
    return 'Done' if stderr == "" else stderr

@shared_task(bind=True)
def prepare_summary(self, mid, task_id=None):
    progress_recorder = ProgressRecorder(self)
    progress_recorder.set_progress(1,2)
    meeting = Meeting.objects.get(pk=mid)
    summarizer_path = os.path.join(settings.BASE_DIR, 'summarization.py')
    process = Popen(['python', summarizer_path, '--filename', meeting.audio.path], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    stdout, stderr = stdout.decode('utf-8').strip(), stderr.decode('utf-8').strip()
    print(stdout)
    print(stderr)

    try:
        output = ast.literal_eval(stdout)
        output["mid"] = meeting
    except:
        traceback.print_exc()
        output = {"abs_summary": stderr ,"ext_summary": stderr, "mid": meeting}

    summary = Summary(**output)
    summary.save()

    wavs = [os.path.join(settings.BASE_DIR, 'audio_chunks',w) for w in os.listdir('audio_chunks') if w.endswith('.wav')]
    for wav in wavs:
        os.remove(wav)
    progress_recorder.set_progress(2,2)
    return stdout if stdout != "" else stderr
