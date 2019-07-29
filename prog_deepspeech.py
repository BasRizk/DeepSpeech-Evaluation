#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing performance of Mozilla Deepspeech on different devices

Created on Tue Mar 26 14:32:43 2019

@author: ironbas8
"""

from deepspeech import Model
from os import path, makedirs
from jiwer import wer    
import soundfile as sf
import sys

from timeit import default_timer as timer
from utils import get_platform_id, document_machine
from utils import prepare_pathes, get_metafiles_pathes
 
#DEEPSPEECH_VERSION="0.4.1"
#DEEPSPEECH_VERSION="0.5.0"
#DEEPSPEECH_VERSION="0.5.0+6_gram_lm"
DEEPSPEECH_VERSION="0.5.1"

TEST_PATH="tests/LibriSpeech/test-clean"
#TEST_PATH="tests/LibriSpeech/test-other"
#TEST_PATH="tests/iisys"

USING_GPU = False
USE_LANGUAGE_MODEL = True
USE_TFLITE = False
USE_MEMORY_MAPPED_MODEL = True
VERBOSE = True
assert(path.exists(TEST_PATH))

try:
    TEST_CORPUS = TEST_PATH.split("/")[1]
    if TEST_CORPUS.lower() == "librispeech":
        TEST_CORPUS += "_" + TEST_PATH.split("/")[2]
except:
    print("WARNING: Path 2nd index does not exist.\n")

if  TEST_CORPUS == "iisys":
    IS_TSV = True
    IS_RECURSIVE_DIRECTORIES = False
else:
    IS_TSV = False
    IS_RECURSIVE_DIRECTORIES = True
    
if IS_TSV:
    TS_INPUT = "tsv"
    AUDIO_INPUT = "wav"
else:
    TS_INPUT = "txt"
    AUDIO_INPUT = "wav"

try:
    if TEST_PATH.split("/")[2] == "Sprecher":
        AUDIO_INPUT="flac"
except:
    print("WARNING: Path 3rd index does not exist.\n")
##############################################################################
# ------------------------Documenting Machine ID
##############################################################################

platform_id = get_platform_id()

if USE_LANGUAGE_MODEL:
    platform_id += "_use_lm"
    
if USE_MEMORY_MAPPED_MODEL and not USE_TFLITE:
    platform_id += "_use_pbmm"
elif USE_TFLITE:
    platform_id += "_use_tflite"
if USING_GPU:
    platform_id += "_use_gpu"

if TEST_CORPUS:
    platform_id = TEST_CORPUS + "_" + AUDIO_INPUT + "_" + platform_id
    

    
platform_meta_path = "logs/v" + DEEPSPEECH_VERSION + "/" + platform_id


    
if not path.exists(platform_meta_path):
    makedirs(platform_meta_path)
    
document_machine(platform_meta_path, USING_GPU)

##############################################################################
# ------------------------------Preparing pathes
##############################################################################

log_filepath, benchmark_filepath = get_metafiles_pathes(platform_meta_path)

test_directories = prepare_pathes(TEST_PATH, recursive = IS_RECURSIVE_DIRECTORIES)
audio_pathes = list()
text_pathes = list()
for d in test_directories:
    audio_pathes.append(prepare_pathes(d, AUDIO_INPUT, recursive = False))
    text_pathes.append(prepare_pathes(d, TS_INPUT, recursive = False))
audio_pathes.sort()
text_pathes.sort()    

##############################################################################
# ----------------------------- Model Loading 
##############################################################################
log_file = open(log_filepath, "w")

# These constants control the beam search decoder
# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500
# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_ALPHA = 0.75
# The beta hyperparameter of the CTC decoder. Word insertion bonus.
LM_BETA = 1.85

# These constants are tied to the shape of the graph used
# Number of MFCC features to use
N_FEATURES = 26
# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9


output_graph_path = "models/v" + DEEPSPEECH_VERSION + "/output_graph"
if USE_MEMORY_MAPPED_MODEL and not USE_TFLITE:
    print("Using MEMORY MAPPED 'pbmm' model.\n")
    output_graph_path += ".pbmm"
elif USE_TFLITE:
    print("Using TF LITE 'tflite' model.\n")
    output_graph_path += ".tflite"
else:
    print("Using Regular 'pb' model.\n")
    output_graph_path += ".pb"

alphabet_path = "models/v" + DEEPSPEECH_VERSION + "/alphabet.txt"
lm_path = "models/v" + DEEPSPEECH_VERSION + "/lm.binary"
trie_path = "models/v" + DEEPSPEECH_VERSION + "/trie"

print('Loading inference model from files {}'.format(output_graph_path),
          file=sys.stderr)
log_file.write('Loading inference model from files {}'.format(output_graph_path))
inf_model_load_start = timer()
ds = Model(output_graph_path,
           N_FEATURES,
           N_CONTEXT,
           alphabet_path,
           BEAM_WIDTH)
inf_model_load_end = timer() - inf_model_load_start
print('Loaded inference model in {:.3}s.'.format(inf_model_load_end))
log_file.write('Loaded inference model in {:.3}s.'.format(inf_model_load_end))

if USE_LANGUAGE_MODEL:
    print('Loading language model from files {} {}'.format(alphabet_path, trie_path),
          file=sys.stderr)
    log_file.write('Loading language model from files {} {}'.format(alphabet_path, trie_path))
    lm_load_start = timer()
    ds.enableDecoderWithLM(alphabet_path, lm_path, trie_path,
                           LM_ALPHA, LM_BETA)
    lm_load_end = timer() - lm_load_start
    print('Loaded language model in {:.3}s.'.format(lm_load_end))
    log_file.write('Loaded language model in {:.3}s.'.format(lm_load_end))

##############################################################################
# ---Running the DeepSpeech STT Engine by running through the audio files
##############################################################################
processed_data = "filename,length(sec),proc_time(sec),wer,actual_text,processed_text\n"
avg_wer = 0
avg_proc_time = 0
num_of_audiofiles = len([item for sublist in audio_pathes for item in sublist])
current_audio_number = 1
all_text_pathes = [item for sublist in text_pathes for item in sublist]
for text_path in all_text_pathes:
    audio_transcripts = open(text_path, 'r').readlines()
    for sample in audio_transcripts:    
        sample_dir = "/".join(text_path.split("/")[:-1])
        if IS_TSV:
            sample_cut = sample[:-1].split("\t")
        else:
            sample_cut = sample[:-1].split(" ")
        sample_audio_path = sample_dir + "/" + sample_cut[0] + "." + AUDIO_INPUT
        sample_transcript = sample_cut[1:]
        
        print("\n=> Progress = " + "{0:.2f}".format((current_audio_number/num_of_audiofiles)*100) + "%\n" )
        current_audio_number+=1

        audio, fs = sf.read(sample_audio_path, dtype='int16')
        audio_len = len(audio)/fs 
            
        print('Running inference.', file=sys.stderr)
        inference_start = timer()
        processed_text = ds.stt(audio, fs)
        inference_end = timer() - inference_start
        print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_len))       
        proc_time = round(inference_end, 3)

        # Processing WORD ERROR RATE (WER)
        actual_text = " ".join(sample_transcript).lower()
        current_wer = wer(actual_text, processed_text, standardize=True)
        current_wer = round(current_wer,3)
        
        # Accumlated data
        avg_proc_time += (proc_time/(audio_len))
        avg_wer += current_wer
        
        progress_row = sample_audio_path + "," + str(audio_len) + "," + str(proc_time)  + "," +\
                        str(current_wer) + "," + actual_text + "," + processed_text
                         
        if(VERBOSE):
            print("# Audio number " + str(current_audio_number) + "/" + str(num_of_audiofiles) +"\n" +\
                  "# File (" + sample_audio_path + "):\n" +\
                  "# - " + str(audio_len) + " seconds long.\n"+\
                  "# - actual    text: '" + actual_text + "'\n" +\
                  "# - processed text: '" + processed_text + "'\n" +\
                  "# - processed in "  + str(proc_time) + " seconds.\n"
                  "# - WER = "  + str(current_wer) + "\n")
                  
        log_file.write("# Audio number " + str(current_audio_number) + "/" + str(num_of_audiofiles) +"\n" +\
              "# File (" + sample_audio_path + "):\n" +\
              "# - " + str(audio_len) + " seconds long.\n"+\
              "# - actual    text: '" + actual_text + "'\n" +\
              "# - processed text: '" + processed_text + "'\n" +\
              "# - processed in "  + str(proc_time) + " seconds.\n"
              "# - WER = "  + str(current_wer) + "\n")
        
                  
        processed_data+= progress_row + "\n"

##############################################################################
# ---------------Finalizing processed data and Saving Logs
##############################################################################
avg_proc_time /= current_audio_number
avg_wer /= current_audio_number
if(VERBOSE):
    print("Avg. Proc. time (sec/second of audio) = " + str(avg_proc_time) + "\n" +\
          "Avg. WER = " + str(avg_wer))
log_file.write("Avg. Proc. time/sec = " + str(avg_proc_time) + "\n" +\
          "Avg. WER = " + str(avg_wer))
log_file.close()
processed_data+= "AvgProcTime (sec/second of audio)," + str(avg_proc_time) + ",,,," + "\n"
processed_data+= "AvgWER," + str(avg_wer) + ",,,,,"+ "\n"


with open(benchmark_filepath, 'w') as f:
    for line in processed_data:
        f.write(line)
    
    
