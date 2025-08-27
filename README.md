# Careful Whisper

> No careless whispers, only words colored with confidence.

A [whisper](https://github.com/openai/whisper)-based, lightweight transcription pipeline for processing audio and/or video files in batch mode, generating accurate transcriptions with **color-coded confidence scores** to indicate the reliability of each transcribed segment. For the demo of whisper's capability, visit [replicate.com/openai/whisper](https://replicate.com/openai/whisper).

To start up, install the dependencies from the file `requirements.txt`. Note that on linux, you need an additional step `sudo apt install ffmpeg`. You can run `python pipe.py` to start the pipeline after you modify your folder name and model preference in `pipe.py`.

![demo_picture](assets/demo.png)

## Future Plan
- [x] Support smarter paragraph splitting
- [ ] Support configuring input, output, and model preference through argument parsing
- [ ] Add a test program for demonstration
