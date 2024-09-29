---
title: Visual Story Generator
emoji: 📉
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# Visual Story Generator 📉

## Overview
The Visual Story Generator is an innovative application that generates engaging short stories based on user-uploaded images. By leveraging state-of-the-art AI models, the application interprets images, extracts meaningful context, and crafts narratives that enhance the storytelling experience.

## Features
- **Image Input**: Users can upload images that the model will analyze to generate a corresponding story.
- **Dynamic Character and Setting**: Users can input characters and settings, with defaults provided if not specified.
- **Language Support**: The application supports both English and Arabic story generation.
- **Text-to-Speech Integration**: Users can listen to the generated stories.

## Models Used
1. **BLIP (Bootstrapping Language-Image Pre-training)**:
   - Used for generating captions from images.
   - **Hugging Face Model**: [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)

2. **GPT-2**:
   - Utilized for generating creative stories based on the provided captions, characters, and settings.
   - **Hugging Face Model**: [gpt2](https://huggingface.co/gpt2)

3. **Translation Model**:
   - Used for translating English stories into Arabic.
   - **Hugging Face Model**: [Helsinki-NLP/opus-mt-en-ar](https://huggingface.co/Helsinki-NLP/opus-mt-en-ar)

## Pipeline Explanations
- **Image Processing**: Uploaded images are processed using the BLIP model to generate captions that describe the content of the image. This step provides context for story generation.
  
- **Story Generation**: The generated captions, along with user-defined characters and settings, are fed into the GPT-2 model to create a coherent story. The model integrates the captions seamlessly into the narrative.

- **Translation**: If the user selects Arabic, the generated story is translated using the translation model, allowing Arabic-speaking users to enjoy the stories.

## Expected Outputs
- **Image Descriptions**: Captions generated from the uploaded images provide a summary of the visual content.
- **Generated Story**: A short story that incorporates the image captions, user-defined characters, and settings.
- **Audio Narration**: An audio file of the generated story read aloud, enhancing accessibility.

## Special Measures for Arabic Language Support
The application incorporates a translation model to ensure that generated stories can be effectively translated into Arabic. This allows for a broader audience reach and enhances accessibility for Arabic-speaking users.

## Walkthrough Video
For a comprehensive guide on how to use the Visual Story Generator, check out our [YouTube Walkthrough](https://youtu.be/OEVJUImbdK8).

## License
This project is licensed under the Apache License 2.0.

## Getting Started
To run this application locally, ensure you have the following Python packages installed:

```plaintext
gradio
transformers
torch
Pillow
gtts
