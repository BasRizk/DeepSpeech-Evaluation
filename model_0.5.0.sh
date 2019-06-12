#!/usr/bin/env bash

wget --no-check-certificate https://github.com/mozilla/DeepSpeech/releases/download/v0.5.0/deepspeech-0.5.0-models.tar.gz
tar -xzvf deepspeech-0.5.0-models.tar.gz -C v0.5.0
rm -f deepspeech-0.5.0-models.tar.gz
mkdir models
mv v0.5.0 models
echo 'Mozilla deepspeech 0.5.0 pre-trained model imported successfully."
