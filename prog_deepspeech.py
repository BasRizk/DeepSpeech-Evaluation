#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing performance of Mozilla Deepspeech on different devices

Created on Tue Mar 26 14:32:43 2019

@author: ironbas8
"""

from deepspeech import Model
from os import listdir, path, makedirs
from jiwer import wer    
import soundfile as sf
import time

import platform, os
 
is_global_directories = True
gpu_support = False
verbose = True
##############################################################################
# ------------------------Documenting Machine ID
##############################################################################

def cpu_info():
    if platform.system() == 'Windows':
        return platform.processor()
    elif platform.system() == 'Darwin':
        command = '/usr/sbin/sysctl -n machdep.cpu.brand_string'
        return os.popen(command).read().strip()
    elif platform.system() == 'Linux':
        command = 'cat /proc/cpuinfo'
        return os.popen(command).read().strip()
    return 'platform not identified'

def gpu_info():
    if platform.system() == 'Linux':
        command = 'nvidia-smi'
        return os.popen(command).read().strip()
    return 'platform not identified'

localtime = time.strftime("%Y%m%d-%H%M%S")
platform_id = platform.machine() + "_" + platform.system() + "_" +\
                platform.node() + "_" + localtime
platform_meta_path = "logs/" + platform_id

if not path.exists(platform_meta_path):
    makedirs(platform_meta_path)

if(gpu_support):
    with open(os.path.join(platform_meta_path,"gpu_info.txt"), 'w') as f:
        f.write(gpu_info())
else:
    with open(os.path.join(platform_meta_path,"cpu_info.txt"), 'w') as f:
        f.write(cpu_info())


##############################################################################
# ------------------------------Preparing pathes
##############################################################################
def prepare_pathes(directory, exten = '', global_dir=False):
    updated_pathes = list()
    if(global_dir):
        subdirectories = listdir(directory)
        subdirectories.sort()
        for subdirectory in subdirectories:
            subdirectory = path.join(directory, subdirectory)
            filenames = listdir(subdirectory)
            filenames.sort()
            for filename in filenames:
                if(filename.endswith(exten)):
                    updated_pathes.append(path.join(subdirectory, filename))
            updated_pathes.sort()
    else:
        filenames = listdir(directory)
        for filename in filenames:
            if(filename.endswith(exten)):
                updated_pathes.append(path.join(directory, filename))
    updated_pathes.sort()
    return updated_pathes

localtime = time.strftime("%Y%m%d-%H%M%S")
log_filepath = platform_meta_path  +"/logs_" + localtime + ".txt"
benchmark_filepath = platform_meta_path  +"/deepspeech041_benchmark_ " + localtime + ".csv"
test_directories = prepare_pathes("tests/current_tests")
audio_pathes = list()
text_pathes = list()
for d in test_directories:
    audio_pathes.append(prepare_pathes(d, "flac", global_dir=is_global_directories))
    text_pathes.append(prepare_pathes(d, "txt", global_dir=is_global_directories))
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
processed_data = "filename,length(sec),proc_time(sec),wer,actual_text,processed_text\n"
avg_wer = 0
avg_proc_time = 0
num_of_audiofiles = len([item for sublist in audio_pathes for item in sublist])
current_audio_number = 1
for audio_group, audio_text_group_path in zip(audio_pathes, text_pathes):
    audio_group.sort()
    audio_transcripts = open(audio_text_group_path[0], 'r').readlines()
    for audio_path, audio_transcript in zip(audio_group, audio_transcripts):
        
        print("\n=> Progress = " + "{0:.2f}".format((current_audio_number/num_of_audiofiles)*100) + "%\n" )
        current_audio_number+=1
        
        audio, fs = sf.read(audio_path, dtype='int16')    
        start_proc = time.time()
        processed_text = ds.stt(audio, fs)
        end = time.time()
       
        audio_len = len(audio)/fs 
        proc_time = end-start_proc
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
                         
        if(verbose):
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
if(verbose):
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
    
    
