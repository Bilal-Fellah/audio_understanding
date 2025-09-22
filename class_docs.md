
# SpeechRecognitionForClassification

A Python class for **automatic speech recognition (ASR), text refinement, and classification**.

It supports two transcription backends:

* **Whisper (PyTorch)** → GPU or CPU
* **Whisper.cpp (CLI)** → CPU-efficient, no GPU required

Additionally, it integrates with **Google Gemini** for text refinement and classification.

---

## Features

* 🎙 **Transcription**: Convert audio into text with Whisper or Whisper.cpp
* 📝 **Refinement**: Automatically correct ASR errors while preserving meaning
* 🏷 **Classification**: Assign transcriptions to user-defined categories with Gemini
* 📂 **Batch Processing**: Process entire folders of audio files
* 📑 **JSON Support**: Save, read, and merge transcriptions with dataset mappings

---

## Requirements

* Python 3.9+
* Install dependencies:

```bash
pip install torch openai-whisper google-generativeai
```

For Whisper.cpp backend:

* Build [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
* Ensure `whisper-cli.exe` (Windows) or `./main` (Linux/Mac) is available

---

## Initialization

```python
from speech_recog import SpeechRecognitionForClassification

sr = SpeechRecognitionForClassification(
    audio_folder="./audio",
    classes=["comedy", "business", "cars", "tourism", "stories"],
    audio_mapping_file="json_audio_mapping.json",
    gemini_api_key="YOUR_GEMINI_API_KEY",
    backend="whisper"  # or "whispercpp"
)
```

---

## Basic Usage

### 1. Transcribe a single file

```python
result = sr.generate_transcription("example.wav")
print(result["text"])
```

### 2. Refine transcription

```python
refined = sr.generate_refinement(result["text"])
print(refined)
```

### 3. Classify text

```python
classes = sr.classify_text(refined)
print(classes)  # e.g., ['comedy', 'stories']
```

### 4. Classify a single audio file

```python
pred = sr.classify_audio("example.wav")
print(pred)  # e.g., ['business']
```

### 5. Process an entire folder

```python
results = sr.classify_folder()
print(results)
# [{'audio': 'file1.wav', 'classes': ['comedy']}, {'audio': 'file2.mp3', 'classes': ['cars', 'tourism']}]
```

---

## JSON Integration

### Save transcriptions

```python
files = sr.load_audio_files()
transcriptions = sr.generate_transcriptions(files)
sr.save_transcriptions(transcriptions, "output_transcriptions.json")
```

### Merge with existing mapping file

If your dataset has a JSON mapping like:

```json
[
  {"audio": "file1", "label": "comedy"},
  {"audio": "file2", "label": "business"}
]
```

Run:

```python
sr.add_transcription_to_original_file()
```

This adds transcriptions under `"whisper_transcription"`.

---

## Switching Backends

* **PyTorch Whisper** (default):

```python
backend="whisper"
```

* **Whisper.cpp** (CPU, faster on low-resource machines):

```python
backend="whispercpp"
```

Adjust paths if needed:

```python
sr.generate_transcription(
    "file.wav",
    exe_path=r"C:\path\to\whisper-cli.exe",
    model_path=r"C:\path\to\ggml-medium.bin",
    threads=6
)
```

---

## Example End-to-End Script

```python
sr = SpeechRecognitionForClassification(
    audio_folder="./audio",
    classes=["comedy", "business", "cars", "tourism", "stories"],
    gemini_api_key="YOUR_GEMINI_API_KEY",
    backend="whispercpp"
)

results = sr.classify_folder()
sr.save_to_json(results, "classified_results.json")
```

