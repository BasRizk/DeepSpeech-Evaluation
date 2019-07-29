#!/usr/bin/env bash

mkdir -p tests/LibriSpeech
if [ ! -d "tests/LibriSpeech/test-clean" ]
then
	wget --no-check-certificate http://www.openslr.org/resources/12/test-clean.tar.gz
	tar -xzvf test-clean.tar.gz
	rm -f test-clean.tar.gz

	mv LibriSpeech/test-clean/LibriSpeech/test-clean tests/LibriSpeech/test-clean
	rm -rf test-clean
	echo "LibriSpeech Clean test set imported successfully."
fi

if [ ! -d "tests/LibriSpeech/test-other" ]
then
	wget --no-check-certificate http://www.openslr.org/resources/12/test-other.tar.gz
	tar -xzvf test-other.tar.gz
	rm -rf test-other.tar.gz
	mv LibriSpeech/test-other/LibriSpeech/test-other tests/LibriSpeech/test-other
	rm -rf test-other
	echo "LibriSpeech Other test set imported successfully."
fi
