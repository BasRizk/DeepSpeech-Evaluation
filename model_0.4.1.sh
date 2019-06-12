#!/usr/bin/env bash

wget --no-check-certificate https://github.com/mozilla/DeepSpeech/releases/download/v0.4.1/deepspeech-0.4.1-models.tar.gz
tar -xzvf deepspeech-0.4.1-models.tar.gz -C v0.4.1
rm -f deepspeech-0.4.1-models.tar.gz
mkdir models
mv v0.4.1 models
echo 'Mozilla deepspeech 0.4.1 pre-trained model imported successfully."
