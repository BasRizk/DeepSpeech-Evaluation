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
 

DEEPSPEECH_VERSION="0.5.0"
TEST_PATH="tests/LibriSpeech/test-clean"
IS_GLOBAL_DIRECTORIES = True
USING_GPU = False
USE_LANGUAGE_MODEL = False
VERBOSE = True
AUDIO_INPUT = "flac"
TS_INPUT = "txt"
##############################################################################
# ------------------------Documenting Machine ID
##############################################################################

platform_id = get_platform_id()
platform_meta_path = "logs/" + platform_id

if not path.exists(platform_meta_path):
    makedirs(platform_meta_path)
    
document_machine()

##############################################################################
# ------------------------------Preparing pathes
##############################################################################

log_filepath, benchmark_filepath = get_metafiles_pathes(platform_meta_path)

test_directories = prepare_pathes(TEST_PATH)
audio_pathes = list()
text_pathes = list()
for d in test_directories:
    audio_pathes.append(prepare_pathes(d, AUDIO_INPUT, global_dir=IS_GLOBAL_DIRECTORIES))
    text_pathes.append(prepare_pathes(d, TS_INPUT, global_dir=IS_GLOBAL_DIRECTORIES))
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
    
output_graph_path = "models/v" + DEEPSPEECH_VERSION + "/output_graph.pb"
alphabet_path = "models/v" + DEEPSPEECH_VERSION + "/alphabet.txt"
lm_path = "models/v" + DEEPSPEECH_VERSION + "lm.binary"
trie_path = "models/v" + DEEPSPEECH_VERSION + "trie"

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
for audio_group, audio_text_group_path in zip(audio_pathes, text_pathes):
    audio_group.sort()
    audio_transcripts = open(audio_text_group_path[0], 'r').readlines()
    audio_transcripts.sort()
    for audio_path, audio_transcript in zip(audio_group, audio_transcripts):
        
        print("\n=> Progress = " + "{0:.2f}".format((current_audio_number/num_of_audiofiles)*100) + "%\n" )
        current_audio_number+=1
        
        audio, fs = sf.read(audio_path, dtype='int16')
        audio_len = len(audio)/fs 

        #start_proc = time.time()
        print('Running inference.', file=sys.stderr)
        inference_start = timer()
        processed_text = ds.stt(audio, fs)
        inference_end = timer() - inference_start
        print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_len))
        #end = time.time()
       
        proc_time = inference_end
        proc_time = round(proc_time,3)

    
        # Processing WORD ERROR RATE (WER)
        audio_transcript = audio_transcript[:-1].split(" ")
        actual_text = " ".join(audio_transcript[1:]).lower()
        current_wer = wer(actual_text, processed_text, standardize=True)
        current_wer = round(current_wer,3)
        
        # Accumlated data
        avg_proc_time += (proc_time/(audio_len))
        avg_wer += current_wer
        
        
        audio_path = audio_path.split("/")[-1]
        progress_row = audio_path + "," + str(audio_len) + "," + str(proc_time)  + "," +\
                        str(current_wer) + "," + actual_text + "," + processed_text
                         
        if(VERBOSE):
            print("# File (" + audio_path + "):\n" +\
                  "# - " + str(audio_len) + " seconds long.\n"+\
                  "# - actual    text: '" + actual_text + "'\n" +\
                  "# - processed text: '" + processed_text + "'\n" +\
                  "# - processed in "  + str(proc_time) + " seconds.\n"
                  "# - WER = "  + str(current_wer) + "\n")
                  
        log_file.write("# File (" + audio_path + "):\n" +\
              "# - " + str(audio_len) + " seconds long.\n"+\
              "# - actual    text: '" + actual_text + "'\n" +\
              "# - processed text: '" + processed_text + "'\n" +\
              "# - processed in "  + str(proc_time) + " seconds.\n"
              "# - WER = "  + str(current_wer) + "\n")
        
                  
        processed_data+= progress_row + "\n"

##############################################################################
# ---------------Finalizing processed data and Saving Logs
##############################################################################
avg_proc_time /= num_of_audiofiles
avg_wer /= num_of_audiofiles
if(VERBOSE):
    print("Avg. Proc. time (sec/second of audio) = " + str(avg_proc_time) + "\n" +\
          "Avg. WER = " + str(avg_wer))
log_file.write("Avg. Proc. time/sec = " + str(avg_proc_time) + "\n" +\
          "Avg. WER = " + str(avg_wer))
log_file.close()
processed_data+= "AvgProcTime (sec/second of audio)," + str(avg_proc_time) + "\n"
processed_data+= "AvgWER," + str(avg_wer) + "\n"


with open(benchmark_filepath, 'w') as f:
    for line in processed_data:
        f.write(line)
    
    
