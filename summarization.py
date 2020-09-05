
# coding: utf-8

# In[1]:
import sys
import torch
import argparse
import azure.cognitiveservices.speech as speechsdk
from transformers import pipeline
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
summarizer = pipeline("summarization")

# Creates an instance of a speech config with specified subscription key and service region.
# Replace with your own subscription key and region.
speech_key, service_region = "3021013d1649482f91008c7df0a0d971", "centralindia"
speech_config=speechsdk.SpeechConfig(subscription=speech_key, region=service_region)


# In[2]:
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--filename', type=str)
args = parser.parse_args()

def summarize_pipline(audio, chunks_output_folder='audio_chunks'):
    get_audio_chunks(audio)
    transcipt = transcribe_each_chunk(chunks_output_folder)
    ext_summary = summarize(transcipt)
    abs_sumarry = abs_summarize(transcipt)
    return {"abs_summary": abs_sumarry, "ext_summary": ext_summary}


# In[4]:


import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

# a function that splits the audio file into chunks
# and applies speech recognition
def get_audio_chunks(audio, output_folder='audio_chunks'):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    # open the audio file using pydub
    sound = AudioSegment.from_wav(audio)
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 1000,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = output_folder
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
#     whole_text = ""
    # process each chunk
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")


# In[8]:


def transcribe_each_chunk(folder=r'audio_chunks'):
    final_text = ""
    for i,j, k in os.walk(folder):
        files = k
    for file in files:
        audio_filename = folder+'/'+file
        audio_input = speechsdk.audio.AudioConfig(filename=audio_filename)

        # Creates a recognizer with the given settings
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

    #     print("Recognizing first result...")

        # Starts speech recognition, and returns after a single utterance is recognized. The end of a
        # single utterance is determined by listening for silence at the end or until a maximum of 15
        # seconds of audio is processed.  The task returns the recognition text as result.
        # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
        # shot recognition like command or query.
        # For long-running multi-utterance recognition, use start_continuous_recognition() instead.

        result = speech_recognizer.recognize_once()

        # Checks result.
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            #print(result.text)
            final_text += result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized: {}".format(result.no_match_details), file=sys.stderr)
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech Recognition canceled: {}".format(cancellation_details.reason), file=sys.stderr)
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details), file=sys.stderr)
    return final_text

# In[9]:
def summarize(text):
    length = len(text.split())
    all_tokens = text.split()
    if length > 700:
        sum_text = ""
        start = 0
        end = 700
        whats_left = length - 700

        while end < length:
            new_text = " ".join(all_tokens[start:end])
            # print(len(new_text.split()))
            summary = summarizer(new_text, min_length=5, model='facebook/bart-large-cnn')
            # print(summary)
            sum_text += summary[0]['summary_text']
            start += 700

            if whats_left > 700:
                end += 700
                whats_left -= 700

            else:
                end += whats_left

    else:
        summary = summarizer(text, model='facebook/bart-large-cnn')
        sum_text = summary[0]['summary_text']

    return sum_text

def abs_summarize(text):
  src_text = [text]
  model_name = 'google/pegasus-xsum'
  torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
  tokenizer = PegasusTokenizer.from_pretrained(model_name)
  model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
  batch = tokenizer.prepare_seq2seq_batch(src_text, truncation=True, padding='longest').to(torch_device)
  translated = model.generate(**batch)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text[0]


# In[10]:
summary = summarize_pipline(args.filename)
#print("================= Summary =============================")
print(summary)
