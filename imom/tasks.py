from __future__ import absolute_import, unicode_literals

import ast
import os
from celery import shared_task
from django.conf import settings
from subprocess import Popen, PIPE
from summarization import summarize_pipline
from .models import Speaker, Meeting, Transcript
from celery.utils.log import get_task_logger

@shared_task
def transcript_summary(mid):
    meeting = Meeting.objects.get(pk=mid)
    pipeline_path = os.path.join(settings.BASE_DIR, 'transcript','pipeline.py')
    process = Popen(['python', pipeline_path, '--filename', meeting.audio.path], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode('utf-8'))
    print(stderr.decode('utf-8'))
    output = stdout.decode('utf-8').strip()
    output = ast.literal_eval(output)
    
    for speaker in output.keys():
        s = Speaker(name='speaker %d' % speaker, mid=meeting)
        s.save()
        for t in output[speaker]:
            entry = Transcript(timestamp=t[0],text=t[1],sid=s,mid=meeting)
            entry.save()

    meeting.summary = summarize_pipline(meeting.audio.path)
    meeting.save()
    