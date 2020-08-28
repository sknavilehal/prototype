from __future__ import absolute_import, unicode_literals

import ast
import os
from celery import shared_task
from django.conf import settings
from subprocess import Popen, PIPE
from .models import Speaker, Meeting, Transcript
from celery.utils.log import get_task_logger

@shared_task
def prepare_transcript(mid):
    meeting = Meeting.objects.get(pk=mid)
    pipeline_path = os.path.join(settings.BASE_DIR, 'transcript','pipeline.py')
    process = Popen(['python', pipeline_path, '--filename', meeting.audio.path], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    output = stdout.decode('utf-8').strip()
    output = ast.literal_eval(output)
    
    for speaker in output.keys():
        s = Speaker(name='speaker %d' % speaker, mid=meeting)
        s.save()
        for t in output[speaker]:
            entry = Transcript(timestamp=t[0],text=t[1],sid=s,mid=meeting)
            entry.save()
    
    meeting.summary = 'done'
    meeting.save()

@shared_task
def update_meeting(mid):
    meeting = Meeting.objects.get(pk=mid)
    meeting.name = 'changed'
    meeting.save()