#!/usr/bin/env bash

wget --no-check-certificate http://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzvf test-clean.tar.gz
rm -f test-clean.tar.gz
mkdir tests
mv test-clean tests
rm -f test-clean
echo 'LibriSpeech Clean test set imported successfully."
