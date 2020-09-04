from __future__ import absolute_import, unicode_literals

import ast
import os
import traceback
from celery import shared_task
from django.conf import settings
from subprocess import Popen, PIPE
from .models import Speaker, Meeting, Transcript
from celery.utils.log import get_task_logger

@shared_task
def transcript_summary(mid):
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

    summarizer_path = os.path.join(settings.BASE_DIR, 'summarization.py')
    process = Popen(['python', summarizer_path, '--filename', meeting.audio.path], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    stdout, stderr = stdout.decode('utf-8').strip(), stderr.decode('utf-8').strip()
    print(stdout)
    print(stderr)
    meeting.summary = stdout
    meeting.save()

    wavs = [os.path.join(settings.BASE_DIR,w) for w in os.listdir() if w.endswith('.wav')]
    wavs += [os.path.join(settings.BASE_DIR, 'audio_chunks',w) for w in os.listdir('audio_chunks') if w.endswith('.wav')]

    for wav in wavs:
        os.remove(wav)
    