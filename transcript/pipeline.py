#!/usr/bin/env python
import argparse
import sys
import os
sys.path.append('transcript/ghostvlad')
import model as spkModel
import toolkits
import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment
import numpy as np
import uisrnn
import librosa
# sys.path.append('visualization')
#from viewer import PlotDiar

# Creates an instance of a speech config with specified subscription key and service region.
# Replace with your own subscription key and region.
speech_key, service_region = "3021013d1649482f91008c7df0a0d971", "centralindia"
speech_config = speechsdk.SpeechConfig(
    subscription=speech_key, region=service_region)


def pipeline(audio):
    timestamps = dia_audio(audio)
    output = asr(audio, timestamps)
    return output


"""A demo script showing how to DIARIZATION ON WAV USING UIS-RNN."""

# ===========================================
#        Parse the argument
# ===========================================


def parse_arguments():
    _DEFAULT_OBSERVATION_DIM = 256

    model_parser = argparse.ArgumentParser(
        description='Model configurations.', add_help=False)

    model_parser.add_argument(
        '--observation_dim',
        default=_DEFAULT_OBSERVATION_DIM,
        type=int,
        help='The dimension of the embeddings (e.g. d-vectors).')

    model_parser.add_argument(
        '--rnn_hidden_size',
        default=512,
        type=int,
        help='The number of nodes for each RNN layer.')
    model_parser.add_argument(
        '--rnn_depth',
        default=1,
        type=int,
        help='The number of RNN layers.')
    model_parser.add_argument(
        '--rnn_dropout',
        default=0.2,
        type=float,
        help='The dropout rate for all RNN layers.')
    model_parser.add_argument(
        '--transition_bias',
        default=None,
        type=float,
        help='The value of p0, corresponding to Eq. (6) in the '
        'paper. If the value is given, we will fix to this value. If the '
        'value is None, we will estimate it from training data '
        'using Eq. (13) in the paper.')
    model_parser.add_argument(
        '--crp_alpha',
        default=1.0,
        type=float,
        help='The value of alpha for the Chinese restaurant process (CRP), '
        'corresponding to Eq. (7) in the paper. In this open source '
        'implementation, currently we only support using a given value '
        'of crp_alpha.')
    model_parser.add_argument(
        '--sigma2',
        default=None,
        type=float,
        help='The value of sigma squared, corresponding to Eq. (11) in the '
        'paper. If the value is given, we will fix to this value. If the '
        'value is None, we will estimate it from training data.')
    model_parser.add_argument(
        '--verbosity',
        default=2,
        type=int,
        help='How verbose will the logging information be. Higher value '
        'represents more verbose information. A general guideline: '
        '0 for errors; 1 for finishing important steps; '
        '2 for finishing less important steps; 3 or above for debugging '
        'information.')

    # training configurations
    training_parser = argparse.ArgumentParser(
        description='Training configurations.', add_help=False)

    training_parser.add_argument(
        '--optimizer',
        '-o',
        default='adam',
        choices=['adam'],
        help='The optimizer for training.')
    training_parser.add_argument(
        '--learning_rate',
        '-l',
        default=1e-5,
        type=float,
        help='The leaning rate for training.')
    training_parser.add_argument(
        '--learning_rate_half_life',
        '-hl',
        default=0,
        type=int,
        help='The half life of the leaning rate for training. If this value is '
        'positive, we reduce learning rate by half every this many '
        'iterations during training. If this value is 0 or negative, '
        'we do not decay learning rate.')
    training_parser.add_argument(
        '--train_iteration',
        '-t',
        default=20000,
        type=int,
        help='The total number of training iterations.')
    training_parser.add_argument(
        '--batch_size',
        '-b',
        default=10,
        type=int,
        help='The batch size for training.')
    training_parser.add_argument(
        '--num_permutations',
        default=10,
        type=int,
        help='The number of permutations per utterance sampled in the training '
        'data.')
    training_parser.add_argument(
        '--sigma_alpha',
        default=1.0,
        type=float,
        help='The inverse gamma shape for estimating sigma2. This value is only '
        'meaningful when sigma2 is not given, and estimated from data.')
    training_parser.add_argument(
        '--sigma_beta',
        default=1.0,
        type=float,
        help='The inverse gamma scale for estimating sigma2. This value is only '
        'meaningful when sigma2 is not given, and estimated from data.')
    training_parser.add_argument(
        '--regularization_weight',
        '-r',
        default=1e-5,
        type=float,
        help='The network regularization multiplicative.')
    training_parser.add_argument(
        '--grad_max_norm',
        default=5.0,
        type=float,
        help='Max norm of the gradient.')
    training_parser.add_argument(
        '--enforce_cluster_id_uniqueness',
        default=True,
        type=bool,
        help='Whether to enforce cluster ID uniqueness across different '
        'training sequences. Only effective when the first input to fit() '
        'is a list of sequences. In general, assume the cluster IDs for two '
        'sequences are [a, b] and [a, c]. If the `a` from the two sequences '
        'are not the same label, then this arg should be True.')

    # inference configurations
    inference_parser = argparse.ArgumentParser(
        description='Inference configurations.', add_help=False)

    inference_parser.add_argument(
        '--beam_size',
        '-s',
        default=10,
        type=int,
        help='The beam search size for inference.')
    inference_parser.add_argument(
        '--look_ahead',
        default=1,
        type=int,
        help='The number of look ahead steps during inference.')
    inference_parser.add_argument(
        '--test_iteration',
        default=2,
        type=int,
        help='During inference, we concatenate M duplicates of the test '
        'sequence, and run inference on this concatenated sequence. '
        'Then we return the inference results on the last duplicate as the '
        'final prediction for the test sequence.')

    idk_parser = argparse.ArgumentParser()
    # set up training configuration.
    idk_parser.add_argument('--filename', type=str)
    idk_parser.add_argument('--gpu', default='', type=str)
    idk_parser.add_argument(
        '--resume', default=r'transcript/ghostvlad/pretrained/weights.h5', type=str)
    idk_parser.add_argument('--data_path', default='4persons', type=str)
    # set up network configuration.
    idk_parser.add_argument('--net', default='resnet34s',
                        choices=['resnet34s', 'resnet34l'], type=str)
    idk_parser.add_argument('--ghost_cluster', default=2, type=int)
    idk_parser.add_argument('--vlad_cluster', default=8, type=int)
    idk_parser.add_argument('--bottleneck_dim', default=512, type=int)
    idk_parser.add_argument('--aggregation_mode', default='gvlad',
                        choices=['avg', 'vlad', 'gvlad'], type=str)
    # set up learning rate, training loss and optimizer.
    idk_parser.add_argument('--loss', default='softmax',
                        choices=['softmax', 'amsoftmax'], type=str)
    idk_parser.add_argument('--test_type', default='normal',
                        choices=['normal', 'hard', 'extend'], type=str)

    model_args, _ = model_parser.parse_known_args()
    idk_args, _ = idk_parser.parse_known_args()
    inference_args, _ = inference_parser.parse_known_args()

    return (model_args, idk_args, inference_args)

SAVED_MODEL_NAME = 'transcript/pretrained/saved_model.uisrnn_benchmark'

model_args, args, inference_args = parse_arguments()

def append2dict(speakerSlice, spk_period):
    key = list(spk_period.keys())[0]
    value = list(spk_period.values())[0]
    timeDict = {}
    timeDict['start'] = int(value[0]+0.5)
    timeDict['stop'] = int(value[1]+0.5)
    if(key in speakerSlice):
        speakerSlice[key].append(timeDict)
    else:
        speakerSlice[key] = [timeDict]

    return speakerSlice


# {'1': [{'start':10, 'stop':20}, {'start':30, 'stop':40}], '2': [{'start':90, 'stop':100}]}
def arrangeResult(labels, time_spec_rate):
    lastLabel = labels[0]
    speakerSlice = {}
    j = 0
    for i, label in enumerate(labels):
        if(label == lastLabel):
            continue
        speakerSlice = append2dict(
            speakerSlice, {lastLabel: (time_spec_rate*j, time_spec_rate*i)})
        j = i
        lastLabel = label
    speakerSlice = append2dict(speakerSlice, {lastLabel: (
        time_spec_rate*j, time_spec_rate*(len(labels)))})
    return speakerSlice


def genMap(intervals):  # interval slices to maptable
    slicelen = [sliced[1]-sliced[0] for sliced in intervals.tolist()]
    mapTable = {}  # vad erased time to origin time, only split points
    idx = 0
    for i, sliced in enumerate(intervals.tolist()):
        mapTable[idx] = sliced[0]
        idx += slicelen[i]
    mapTable[sum(slicelen)] = intervals[-1, -1]

    keys = [k for k, _ in mapTable.items()]
    keys.sort()
    return mapTable, keys

def fmtTime(timeInMillisecond):
    millisecond = timeInMillisecond % 1000
    minute = timeInMillisecond//1000//60
    second = (timeInMillisecond-minute*60*1000)//1000
    time = '{}:{:02d}.{}'.format(minute, second, millisecond)
    return time


def load_wav(vid_path, sr):
    wav, _ = librosa.load(vid_path, sr=sr)
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])
    return np.array(wav_output), (intervals/sr*1000).astype(int)


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length,
                          hop_length=hop_length)  # linear spectrogram
    return linear.T


# 0s        1s        2s                  4s                  6s
# |-------------------|-------------------|-------------------|
# |-------------------|
#           |-------------------|
#                     |-------------------|
#                               |-------------------|
def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, embedding_per_second=0.5, overlap_rate=0.5):
    wav, intervals = load_wav(path, sr=sr)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_mag = mag_T

    spec_len = sr/hop_length/embedding_per_second
    spec_hop_len = spec_len*(1-overlap_rate)

    cur_slide = 0.0
    utterances_spec = []

    while(True):  # slide window.
        if(cur_slide + spec_len > time):
            break
        spec_mag = mag_T[:, int(cur_slide+0.5): int(cur_slide+spec_len+0.5)]

        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterances_spec.append(spec_mag)

        cur_slide += spec_hop_len

    return utterances_spec, intervals


def dia_audio(wav_path, embedding_per_second=0.3, overlap_rate=0.33):

    # gpu configuration
    #toolkits.initialize_GPU(args)

    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }

    network_eval = spkModel.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                   num_class=params['n_classes'],
                                                   mode='eval', args=args)
    network_eval.load_weights(args.resume, by_name=True)

    model_args.observation_dim = 512
    uisrnnModel = uisrnn.UISRNN(model_args)
    uisrnnModel.load(SAVED_MODEL_NAME)

    specs, intervals = load_data(
        wav_path, embedding_per_second=embedding_per_second, overlap_rate=overlap_rate)
    mapTable, keys = genMap(intervals)

    feats = []
    for spec in specs:
        spec = np.expand_dims(np.expand_dims(spec, 0), -1)
        v = network_eval.predict(spec)
        feats += [v]

    feats = np.array(feats)[:, 0, :].astype(float)  # [splits, embedding dim]
    predicted_label = uisrnnModel.predict(feats, inference_args)

    time_spec_rate = 1000*(1.0/embedding_per_second) * \
        (1.0-overlap_rate)  # speaker embedding every ?ms
    center_duration = int(1000*(1.0/embedding_per_second)//2)
    speakerSlice = arrangeResult(predicted_label, time_spec_rate)

    for spk, timeDicts in speakerSlice.items():    # time map to orgin wav(contains mute)
        for tid, timeDict in enumerate(timeDicts):
            s = 0
            e = 0
            for i, key in enumerate(keys):
                if(s != 0 and e != 0):
                    break
                if(s == 0 and key > timeDict['start']):
                    offset = timeDict['start'] - keys[i-1]
                    s = mapTable[keys[i-1]] + offset
                if(e == 0 and key > timeDict['stop']):
                    offset = timeDict['stop'] - keys[i-1]
                    e = mapTable[keys[i-1]] + offset

            speakerSlice[spk][tid]['start'] = s
            speakerSlice[spk][tid]['stop'] = e

    for spk, timeDicts in speakerSlice.items():
        ##print('========= ' + str(spk) + ' =========')
        for timeDict in timeDicts:
            s = timeDict['start']
            e = timeDict['stop']
            s = fmtTime(s)  # change point moves to the center of the slice
            e = fmtTime(e)
            #print(s+' ==> '+e)

#     p = PlotDiar(map=speakerSlice, wav=wav_path, gui=True, size=(25, 6))
#     p.draw()
#     p.plot.show()
    return speakerSlice

# if __name__ == '__main__':
#     out = main(r'wavs/dia.wav', embedding_per_second=0.3, overlap_rate=0.35)
#     #print(out)


def transcribe(audio):
    audio_filename = audio
    audio_input = speechsdk.audio.AudioConfig(filename=audio_filename)

    # Creates a recognizer with the given settings
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_input)

    #     #print("Recognizing first result...")

    # Starts speech recognition, and returns after a single utterance is recognized. The end of a
    # single utterance is determined by listening for silence at the end or until a maximum of 15
    # seconds of audio is processed.  The task returns the recognition text as result.
    # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
    # shot recognition like command or query.
    # For long-running multi-utterance recognition, use start_continuous_recognition() instead.

    result = speech_recognizer.recognize_once()

    # Checks result.
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details), file=sys.stderr)
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason), file=sys.stderr)
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details), file=sys.stderr)


# def asr(audio, timestamps):
#     d = dict()
#     a = AudioSegment.from_wav(audio)
#     for speaker in timestamps.keys():
#         i = 1
#         d[speaker] = []
#         for timestamp in timestamps[speaker]:
#             clip = a[timestamp['start']:timestamp['stop']]
#             filename = str(speaker)+"_"+str(i)+".wav"
#             clip.export(filename, format='wav')
#             transcript = transcribe(filename)
#             #print(transcript)
#             t = str(timestamp['start']) + " to " + str(timestamp['stop'])
#             #print(t)
#             d[speaker].append((timestamp['start'], transcript))
#             i += 1
#     return d
def asr(audio, timestamps):
    d = dict()
    a = AudioSegment.from_wav(audio)
    for speaker in timestamps.keys():
        i = 1
        d[speaker] = []
        for timestamp in timestamps[speaker]:
            duration = timestamp['stop'] - timestamp['start']
#             print(duration)
            if duration > 10000:
                j = 0
                transcript = ""
                start = timestamp['start']
                stop = timestamp['start'] + 10000
                while True:
                    clip = a[start:stop]
                    filename = str(speaker)+"_"+str(i)+"_"+str(j)+".wav"
                    clip.export(filename, format='wav')
                    try:
                        transcript += transcribe(filename)
#                         print(transcript)
#                         d[speaker][start] = transcript
                    except Exception:
#                         print("Nothing transcribed")
                        if stop >= timestamp['stop']:break
                    j += 1
                    left = timestamp['stop'] - stop
                    start += 10000
                    if left > 10000:
                        stop += 10000
                    else:
                        stop += left
                if transcript != "":
                    d[speaker].append((timestamp['start'],transcript))

            else:
                clip = a[timestamp['start']:timestamp['stop']] # get the first second of an mp3
                filename = str(speaker)+"_"+str(i)+".wav"
                clip.export(filename, format='wav')
                transcript = transcribe(filename)
                t = str(timestamp['start'])+" to "+str(timestamp['stop'])
                d[speaker].append((timestamp['start'],transcript))
            i += 1
    return d

result = pipeline(args.filename)
if not result: result = {0 : [(0,"No speech could be recognized")]}
print(result)
