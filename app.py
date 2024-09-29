import gradio as gr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import random
from gtts import gTTS
import re

# Load the BLIP model for generating captions
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


# Load GPT-2 model for story generation
story_generator = pipeline("text-generation", model="gpt2")

# Load translation model (English to Arabic)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar")

# Default characters and settings
default_characters = [
    "Ali", "Fatima", "Omar", "Amina", "Zaid", "Layla", "Hassan", "Sara",
    "Yusuf", "Noura", "Khalid", "Rania", "Amir", "Jasmine", "Farah", "Sami",
    "Aisha", "Rami", "Zayn", "Dalia", "Bilal", "Ibtisam", "Mansour", "Afnan",
    "Jamal", "Asma", "Khadija", "Hadi", "Maya", "Samir", "Nabil", "Lina",
    "Tariq", "Yara", "Munir", "Ranya", "Firas", "Nadia", "Alaa", "Nida",
    "Omar", "Dina", "Zein", "Rami", "Yasmin", "Salma", "Jamil", "Khaled"
]

default_settings = [
    "a mystical forest", "a bustling city", "an ancient castle", "a snowy mountain village",
    "a sunny beach", "a dark cave", "a magical kingdom", "a quiet library",
    "a colorful carnival", "a haunted house", "a space station", "a serene garden",
    "a busy marketplace", "a futuristic city", "a pirate ship", "a wild savannah",
    "a snowy tundra", "a tropical island", "an underwater city", "a hidden valley",
    "a dragon's lair", "an enchanted meadow", "a witch's cottage", "a cozy cabin",
    "a bustling train station", "a giant's castle", "a fairy-tale village",
    "a mysterious island", "a historical battlefield", "an abandoned factory",
    "a magical forest glen", "a secret garden", "a royal palace", "a shimmering lagoon",
    "a giant treehouse", "a whimsical tree-lined street", "a rugged mountain range",
    "a starry night sky", "a bright sunny day", "a lively zoo", "an exciting amusement park"
]


def preprocess_image(image):
    return image.convert("RGB").resize((256, 256))


def get_captions(images):
    captions = []
    for img in images:
        processed_img = preprocess_image(img)
        inputs = blip_processor(images=processed_img, return_tensors="pt")
        caption = blip_model.generate(**inputs)
        caption_text = blip_processor.decode(caption[0], skip_special_tokens=True)
        captions.append(caption_text)
    return " ".join(captions)


def clean_up_caption(caption):
    caption = caption.strip().capitalize()
    scene_descriptions = ["scene", "view", "image", "photo", "picture"]
    for word in scene_descriptions:
        caption = caption.replace(word, "").strip()

    # Simplifying specific terms
    caption = caption.replace("a snowy scene", "on a snowy mountain").replace("log cabin", "log cabin").strip()

    return caption


def generate_relevant_setting(captions):
    keywords_to_settings = {
        "castle": "an ancient castle",
        "forest": "a mystical forest",
        "mountain": "a snowy mountain village",
        "beach": "a sunny beach"
    }

    for keyword, setting in keywords_to_settings.items():
        if keyword in captions.lower():
            return setting
    return "a mysterious place"


def integrate_caption_into_story(caption, character, setting):
    cleaned_caption = clean_up_caption(caption)

    # Lists for interaction scenarios
    animals = ["cat", "dog", "bear", "lion", "eagle", "bird", "rabbit", "tiger", "elephant", "fish", "horse", "wolf",
               "deer"]
    beings = ["girl", "boy", "man", "woman", "child", "hero", "princess", "prince", "wizard", "witch", "monster",
              "creature"]
    items = ["book", "sword", "fruit", "food", "technology", "map", "key", "potion", "gem", "tool", "lantern", "shield"]

    if "cabin" in cleaned_caption and "snow" in cleaned_caption:
        return f"In a cozy log cabin on a snowy mountain, there lived a brave character named {character}."
    elif "mountain" in cleaned_caption:
        return f"High up in the mountains, {character} embarked on an adventure."
    elif any(animal in cleaned_caption for animal in animals):
        return f"One day, {character} encountered a {cleaned_caption} that changed everything."
    elif any(being in cleaned_caption for being in beings):
        return f"A {cleaned_caption} approached {character} with a sense of wonder."
    elif "treasure" in cleaned_caption:
        return f"{character} stumbled upon a treasure in the {cleaned_caption}, which held many secrets."
    elif any(item in cleaned_caption for item in items):
        return f"{character} picked up a {cleaned_caption}, which turned out to be very special."
    else:
        return f"{character} was surrounded by {cleaned_caption} during their adventure."


def trim_story(story, max_lines, max_words=None):
    # Split the story into sentences using regex to handle punctuation correctly
    sentences = re.split(r'(?<=[.!?]) +', story.strip())
    trimmed_lines = []
    line_count = 0
    word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        # If max_words is specified, check against it
        if max_words is not None and (word_count + sentence_word_count) > max_words:
            break

        # Check if adding this sentence would exceed max_lines
        if line_count < max_lines:
            trimmed_lines.append(sentence.strip())
            line_count += 1
            word_count += sentence_word_count
        else:
            break


    return ' '.join(trimmed_lines)


def generate_story_from_images(images, story_length, character, setting, tone, language):
    try:
        if isinstance(images, Image.Image):
            images = [images]

        images = [Image.fromarray(image) if not isinstance(image, Image.Image) else image for image in images]
        combined_captions = get_captions(images)

        if not character.strip():
            character = random.choice(default_characters)  # Random character
        if not setting.strip():
            setting = generate_relevant_setting(combined_captions)

        # Set the maximum tokens and lines based on story length
        if story_length == "Short":
            max_new_tokens = 50
            max_lines = 5
        elif story_length == "Medium":
            max_new_tokens = 100
            max_lines = 15
        else:  # Long
            max_new_tokens = 300
            max_lines = 90

        caption_in_story = integrate_caption_into_story(combined_captions, character, setting)

        story_prompt = (
            f"Once upon a time, in {setting}, there was a kind and brave character named {character}. "
            f"{caption_in_story} They had many adventures filled with fun and wonder."
        ) if tone == "Kids Story" else (
            f"In {setting}, a character named {character} faced challenges and deep emotions. "
            f"{caption_in_story} Their journey was filled with tension, suspense, and moments of heartfelt struggle."
        )

        # Check length and truncate if necessary
        if len(story_prompt) > 1024:  # Limit input length to prevent exceeding model limits
            story_prompt = story_prompt[:1024]

        # Generate the story
        story = story_generator(
            story_prompt,
            max_new_tokens=max_new_tokens,
            truncation=True,
            pad_token_id=story_generator.tokenizer.eos_token_id,
            num_return_sequences=1
        )
        generated_story = story[0]['generated_text']
        trimmed_story = trim_story(generated_story, max_lines)

        if language == "Arabic":
            trimmed_story = translator(trimmed_story)[0]['translation_text']

        return combined_captions, trimmed_story

    except Exception as e:
        return str(e), "Error generating story."


def text_to_speech(story_text, language):
    lang = 'ar' if language == "Arabic" else 'en'
    tts = gTTS(text=story_text, lang=lang)
    audio_file = "story.mp3"
    tts.save(audio_file)
    return audio_file


# Create a Gradio interface
with gr.Blocks() as interface:
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Images")

    character_setting_language = gr.Row()
    character_input = gr.Textbox(label="Character Name (Optional)",
                                 placeholder="Enter character name...")
    setting_input = gr.Textbox(label="Setting (Optional)", placeholder="Enter story setting...")
    story_length = gr.Radio(
        choices=["Short", "Medium", "Long"],
        label="Select Story Length",
        value="Medium"  # Default value
    )

    language = gr.Radio(
        choices=["English", "Arabic"],
        label="Select Story Language",
        value="English"  # Default value
    )

    tone = gr.Radio(
        choices=["Kids Story", "Drama"],
        label="Select Story Style/Tone",
        value="Kids Story"  # Default value
    )

    image_descriptions = gr.Textbox(label="Image Descriptions", interactive=False)
    generated_story = gr.Textbox(label="Generated Story", interactive=False)

    submit_button = gr.Button("Generate Story")
    listen_button = gr.Button("Listen to Story")

    submit_button.click(
        fn=generate_story_from_images,
        inputs=[image_input, story_length, character_input, setting_input, tone, language],
        outputs=[image_descriptions, generated_story]
    )

    listen_button.click(
        fn=text_to_speech,
        inputs=[generated_story, language],
        outputs=gr.Audio(label="Story Audio", type="filepath")  # Output audio file
    )

# Launch the app
if __name__ == "__main__":
    interface.launch()
