1. Test .wav files with whisper turbo and compare with mp3 files
 => MP3 files are better in terms of **Time** and **Quality**
 - done in /audio_formats/test_time.ipynb
 
2. Upload missing files to drive

3. Test Whisper Large vs Turbo for GPU
=> Turbo is 3 times faster and provides same (or slightly better) quality

4. Test cpp version Large vs Turbo
=> Turbo is 1.75 times faster and provides same quality

5. do classification with the new models
=> Result looks good

6. test bad quality audio (whisper L gpu vs cpu)
=> looking at the cosine similarity, they provide the same quality
=> Manual checking gives same insight, GPU and CPU results has same quality