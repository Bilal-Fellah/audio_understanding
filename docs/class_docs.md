# VoiceTopicExtraction Documentation

A comprehensive Python pipeline for audio transcription, refinement, and classification using Whisper and Google Gemini AI.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Architecture](#architecture)
6. [API Reference](#api-reference)
7. [Usage Examples](#usage-examples)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)

---

## Overview

**VoiceTopicExtraction** is a Python class that automates the process of:
- **Transcribing** audio files using OpenAI's Whisper (GPU or CPU)
- **Refining** raw transcriptions with Google Gemini AI to correct errors
- **Classifying** refined text into user-defined categories
- **Managing** transcription history in JSON format

This pipeline is particularly optimized for **Algerian Arabic (Darja)** but can work with any language supported by Whisper.

---

## Features

✅ **Dual Transcription Backends**
- GPU-accelerated Whisper (PyTorch)
- CPU-optimized whisper.cpp

✅ **AI-Powered Refinement**
- Corrects spelling and word segmentation
- Fixes place names
- Preserves original meaning

✅ **Multi-Label Classification**
- Supports multiple categories per audio
- Customizable category lists

✅ **History Management**
- Automatically saves all results to JSON
- Includes raw and refined transcriptions

✅ **Batch Processing**
- Process entire folders at once
- Robust error handling per file

---

## Installation

### Prerequisites

```bash
# Core dependencies
pip install torch whisper google-generativeai

# For GPU support (optional)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### whisper.cpp Setup (for CPU mode)

1. Download whisper.cpp from [GitHub](https://github.com/ggerganov/whisper.cpp)
2. Build the executable following their instructions
3. Download a model file (e.g., `ggml-medium.bin`)
4. Note the paths to `whisper-cli.exe` and the model file

---

## Quick Start

### Basic Usage

```python
from voice_topic_extraction import VoiceTopicExtraction

# Initialize the pipeline
extractor = VoiceTopicExtraction(
    audio_folder="./my_audio_files",
    classes=["politics", "sports", "technology", "weather"],
    gemini_api_key="your-gemini-api-key-here"
)

# Process all audio files in the folder
results = extractor.classify_folder(device="cpu")

# View results
for result in results:
    print(f"{result['audio']}: {result['classes']}")
```

### Single File Processing

```python
# Classify a single audio file
classes = extractor.classify_audio(
    audio_path="conversation.mp3",
    device="cuda",  # Use GPU
    transcription_model="medium"
)

print(f"Detected topics: {classes}")
```

---

## Architecture

### Pipeline Flow

```
┌─────────────┐
│ Audio File  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Whisper/whisper.cpp │ ← Transcription
│  (GPU or CPU)       │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Raw Transcription   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Google Gemini AI    │ ← Refinement
│  (Spell Check)      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Refined Text        │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Google Gemini AI    │ ← Classification
│  (Topic Detection)  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Topic Classes       │
│ + Save to JSON      │
└─────────────────────┘
```

### Component Responsibilities

| Component | Purpose |
|-----------|---------|
| **Whisper** | Converts audio to raw text |
| **Gemini Refinement** | Corrects transcription errors |
| **Gemini Classification** | Assigns topic categories |
| **JSON Manager** | Stores complete history |

---

## API Reference

### Class: `VoiceTopicExtraction`

#### Constructor

```python
VoiceTopicExtraction(
    audio_folder: str,
    classes: List[str] = None,
    audio_mapping_file: str = None,
    gemini_api_key: str = None,
    backend: str = "whisper"
)
```

**Parameters:**
- `audio_folder`: Path to directory containing audio files
- `classes`: List of category labels for classification
- `audio_mapping_file`: JSON file path (default: `"json_audio_mapping.json"`)
- `gemini_api_key`: Google Gemini API key
- `backend`: Backend identifier for logging

---

### Core Methods

#### `classify_audio()`

Process a single audio file through the complete pipeline.

```python
classify_audio(
    audio_path: str,
    device: str = "cpu",
    threads: int = 4,
    transcription_model: str = None,
    exe_path: str = "...",  # Path to whisper-cli.exe
    model_path: str = "..."  # Path to ggml model
) -> List[str]
```

**Returns:** List of predicted category labels

---

#### `classify_folder()`

Process all audio files in a folder.

```python
classify_folder(
    folder_path: str = None,
    device: str = "cpu",
    transcription_model: str = None,
    threads: int = 4
) -> List[Dict[str, Any]]
```

**Returns:** List of `{"audio": filename, "classes": [...]}`

---

#### `generate_transcription()`

Transcribe a single audio file.

```python
generate_transcription(
    audio_file: str,
    device: str = "cpu",
    model_name: str = "medium",
    exe_path: str = "...",
    model_path: str = "...",
    threads: int = 4,
    language: str = "ar"
) -> Dict[str, str]
```

**Returns:** `{"text": "transcription..."}`

---

#### `generate_refinement()`

Refine a transcription with AI.

```python
generate_refinement(
    text: str,
    model: str = "gemini-2.0-flash",
    temperature: float = 0.5
) -> str
```

**Returns:** Refined text string

---

#### `classify_text()`

Classify text into categories.

```python
classify_text(
    text: str,
    language: str = "Algerian Arabic"
) -> List[str]
```

**Returns:** List of category labels

---

## Usage Examples

### Example 1: GPU Transcription with Custom Categories

```python
extractor = VoiceTopicExtraction(
    audio_folder="./interviews",
    classes=["education", "healthcare", "economy", "culture"],
    gemini_api_key="YOUR_API_KEY"
)

# Use GPU for faster processing
results = extractor.classify_folder(
    device="cuda",
    transcription_model="large"  # Higher accuracy
)
```

### Example 2: CPU Processing with whisper.cpp

```python
extractor = VoiceTopicExtraction(
    audio_folder="./podcasts",
    classes=["news", "entertainment", "science"],
    gemini_api_key="YOUR_API_KEY"
)

results = extractor.classify_folder(
    device="cpu",
    threads=8  # Use 8 CPU threads
)
```

### Example 3: Custom whisper.cpp Paths

```python
classes = extractor.classify_audio(
    audio_path="meeting.wav",
    device="cpu",
    exe_path="/custom/path/whisper-cli.exe",
    model_path="/models/ggml-large.bin",
    threads=6
)
```

### Example 4: Direct Transcription Only

```python
# Just transcribe without classification
result = extractor.generate_transcription(
    audio_file="speech.mp3",
    device="cuda",
    model_name="medium",
    language="ar"
)

print(result["text"])
```

### Example 5: Refine Existing Text

```python
raw_text = "هذا نص فيه أخطاء كثيرة"
refined = extractor.generate_refinement(raw_text)
print(refined)
```

---

## Configuration

### Supported Audio Formats

- `.mp3`
- `.wav`
- `.webm`

### Whisper Model Sizes

| Model | Parameters | VRAM | Speed |
|-------|-----------|------|-------|
| tiny | 39M | ~1GB | Fastest |
| base | 74M | ~1GB | Fast |
| small | 244M | ~2GB | Balanced |
| medium | 769M | ~5GB | Accurate |
| large | 1550M | ~10GB | Most Accurate |

### Language Codes

Common language codes for the `language` parameter:
- `"ar"` - Arabic
- `"en"` - English
- `"fr"` - French

[Full list](https://github.com/openai/whisper#available-models-and-languages)

---

## JSON Output Format

The history file contains entries like this:

```json
[
  {
    "audio": "interview_01.mp3",
    "transcription": "raw whisper output...",
    "transcription_refined": "corrected text...",
    "classes": ["politics", "economy"],
    "transcription_engine": "cuda - medium"
  }
]
```

---

## Troubleshooting

### Issue: "Gemini API key is missing"

**Solution:** Ensure you pass `gemini_api_key` to the constructor or call `configure_gemini(api_key="...")` before classification.

### Issue: whisper.cpp not found

**Solution:** Verify the `exe_path` points to a valid `whisper-cli.exe` file. Check the path exists:

```python
import os
print(os.path.exists(r"C:\path\to\whisper-cli.exe"))
```

### Issue: CUDA out of memory

**Solution:** 
- Use a smaller Whisper model (`"small"` or `"base"`)
- Switch to CPU mode: `device="cpu"`
- Close other GPU applications

### Issue: Classification returns unexpected format

**Solution:** The code now handles markdown code blocks. If issues persist, check that `self.classes` is properly set.

### Issue: Empty transcription

**Solution:**
- Verify audio file is not corrupted
- Check language parameter matches audio language
- Ensure audio has clear speech content

---

## Logical Fixes Applied

### 1. **Default Model Handling**
**Problem:** `transcription_model` could be `None`, causing errors.