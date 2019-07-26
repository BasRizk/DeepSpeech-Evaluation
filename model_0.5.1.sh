#!/usr/bin/env bash

wget --no-check-certificate https://github.com/mozilla/DeepSpeech/releases/download/v0.5.1/deepspeech-0.5.1-models.tar.gz
mkdir v0.5.1 && tar -xzvf deepspeech-0.5.1-models.tar.gz -C v0.5.1 --strip-components 1
rm -f deepspeech-0.1.0-models.tar.gz
if [ ! -d models ]
then
    mkdir models
else
   echo "models dir already exists"
fi
mv v0.5.1 models
echo "Mozilla deepspeech 0.5.1 pre-trained model imported successfully."
