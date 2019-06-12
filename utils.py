# -*- coding: utf-8 -*-

from os import listdir, path
import platform, os
import time

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


def get_platform_id():
    localtime = time.strftime("%Y%m%d-%H%M%S")
    return platform.machine() + "_" + platform.system() + "_" +\
                platform.node() + "_" + localtime
            

def document_machine(platform_meta_path, USING_GPU):
    if(USING_GPU):
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

def get_meta_pathes(platform_meta_path):
    localtime = time.strftime("%Y%m%d-%H%M%S")
    log_filepath = platform_meta_path  +"/logs_" + localtime + ".txt"
    benchmark_filepath = platform_meta_path  +"/deepspeech041_benchmark_ " + localtime + ".csv"
    return log_filepath, benchmark_filepath