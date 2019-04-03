#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing performance of Mozilla Deepspeech on different devices

Created on Tue Mar 26 14:32:43 2019

@author: ironbas8
"""

from deepspeech import Model
from os import listdir, path
from jiwer import wer    
import soundfile as sf
import time


verbose = True

##############################################################################
# ------------------------------Preparing pathes
##############################################################################
def prepare_pathes(directory, exten = ''):
    updated_pathes = list()
    filenames = listdir(directory)
    for filename in filenames:
        if(filename.endswith(exten)):
            updated_pathes.append(path.join(directory, filename))
    return updated_pathes

localtime = time.strftime("%Y%m%d-%H%M%S")
log_filepath = "logs/logs_" + localtime + ".txt"
directories = prepare_pathes("tests/current_tests")
audio_pathes = list()
text_pathes = list()
for d in directories:
    audio_pathes.append(prepare_pathes(d, "flac"))
    text_pathes.append(prepare_pathes(d, "txt"))
audio_pathes.sort()
text_pathes.sort()    
    
#    audio_pathes = prepare_pathes("tests/audio", "flac")
#    audio_pathes.sort()
#    text_path = prepare_pathes("tests/text", "txt")
#    audio_transcripts = open(text_path[0], 'r').readlines()


output_graph_path = "models/output_graph.pb"
alphabet_path = "models/alphabet.txt"
ds = Model(output_graph_path,
           26,
           9,
           alphabet_path,
           500)


##############################################################################
# ---Running the DeepSpeech STT Engine by running through the audio files
##############################################################################
log_file = open(log_filepath, "w")
processed_data = "filename,length(sec),actual_text,processed_text,proc_time(ms),wer\n"
avg_wer = 0
avg_proc_time = 0
num_of_audiofiles = len([item for sublist in audio_pathes for item in sublist])
current_audio_number = 0
for audio_group, audio_text_group_path in zip(audio_pathes, text_pathes):
    audio_group.sort()
    audio_transcripts = open(audio_text_group_path[0], 'r').readlines()
    for audio_path, audio_transcript in zip(audio_group, audio_transcripts):
    
        print("\n=> Progress = " + "{0:.2f}".format(current_audio_number/num_of_audiofiles) + "%\n" )
        
        audio, fs = sf.read(audio_path, dtype='int16')    
        start_proc = time.time()
        processed_text = ds.stt(audio, fs)
        end = time.time()
       
        audio_len = len(audio)/fs 
        proc_time = end-start_proc
    
        # Processing WORD ERROR RATE (WER)
        audio_transcript = audio_transcript[:-1].split(" ")
        actual_text = " ".join(audio_transcript[1:]).lower()
        current_wer = wer(actual_text, processed_text, standardize=True)
        
        # Accumlated data
        avg_proc_time += proc_time
        avg_wer += current_wer
        
        progress_row = audio_path + "," + str(audio_len) + "," + actual_text +\
                         "," + processed_text + "," + str(proc_time) +\
                         "," + str(current_wer)
                         
        if(verbose):
            print("# File (" + audio_path + "):\n" +\
                  "# - " + str(audio_len) + " seconds long.\n"+\
                  "# - actual    text: '" + actual_text + "'\n" +\
                  "# - processed text: '" + processed_text + "'\n" +\
                  "# - processed in "  + str(proc_time) + " miliseconds.\n"
                  "# - WER = "  + str(current_wer) + "\n")
                  
        log_file.write("# File (" + audio_path + "):\n" +\
              "# - " + str(audio_len) + " seconds long.\n"+\
              "# - actual    text: '" + actual_text + "'\n" +\
              "# - processed text: '" + processed_text + "'\n" +\
              "# - processed in "  + str(proc_time) + " miliseconds.\n"
              "# - WER = "  + str(current_wer) + "\n")
        
                  
        processed_data+= progress_row + "\n"

##############################################################################
# ---------------Finalizing processed data and Saving Logs
##############################################################################
avg_proc_time /= num_of_audiofiles
avg_wer /= num_of_audiofiles
if(verbose):
    print("Avg. Proc. time = " + str(avg_proc_time) + "\n" +\
          "Avg. WER = " + str(avg_wer))
log_file.write("Avg. Proc. time = " + str(avg_proc_time) + "\n" +\
          "Avg. WER = " + str(avg_wer))
log_file.close()
processed_data+= "Avg. Proc. Time," + str(avg_proc_time) + "\n"
processed_data+= "Avg. WER," + str(avg_wer) + "\n"


with open('logs/deepspeech_benchmark.csv', 'w') as f:
    for line in processed_data:
        f.write(line)
    
    
