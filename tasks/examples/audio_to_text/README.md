# Audio to Text

This example demonstrates how to create a multimodal pipeline for processing audio inputs and generating textual outputs using the GraSP framework. It showcases the integration of audio-capable LLMs like Qwen2-Audio-7B for tasks such as audio classification, speech recognition, and audio content analysis.

> **Key Features**:
> `multimodal processing`, `audio classification`, `base64 encoding`, `audio-capable LLMs`, `HuggingFace dataset integration`

## Overview

The audio to text example is designed to:

- **Process audio inputs**: Handle various audio file formats and sources
- **Enable multimodal prompting**: Combine audio with textual instructions
- **Support audio analysis**: Identify sounds, transcribe speech, and analyze audio content
- **Showcase HuggingFace integration**: Stream data from HuggingFace datasets

## Directory Contents

- `graph_config.yaml`: Configuration file defining the workflow for audio processing

## How It Works

1. **Data Loading**: The system loads audio samples from a HuggingFace dataset:
   - Uses the "datasets-examples/doc-audio-1" repository
   - Streams data to minimize memory requirements

2. **Audio Processing**:
   - Audio files are automatically detected and handled
   - Files are converted to base64-encoded data URLs for LLM compatibility
   - Multiple audio formats are supported (.wav, .flac, .mp3, etc.)

3. **LLM Processing**:
   - The Qwen2-Audio-7B model receives both text instructions and the audio data
   - The model analyzes the audio content based on the instructions
   - In this example, it identifies the animal in the audio

4. **Result Collection**:
   - The system captures the LLM's textual analysis
   - Results are structured into standardized output formats

## Example Output

```json
[
    {
        "id": "sample1",
        "audio_url": "data:audio/wav;base64,UklGRuQAAABXQVZFZm10IBAAAAABAAEA...",
        "analysis": "The audio contains the sound of a dog barking."
    },
    {
        "id": "sample2",
        "audio_url": "data:audio/wav;base64,UklGRuQAAABXQVZFZm10IBAAAAABAAEA...",
        "analysis": "The audio contains the sound of a cat meowing."
    }
]
```