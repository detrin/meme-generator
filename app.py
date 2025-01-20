import json
import gradio as gr
from openai import OpenAI
import os
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime
import os.path

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

def get_meme(meme_seed):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are the best meme lord in the world. You are also highly creative, funny and clever."
            },
            {
                "role": "user",
                "content": f"Generate me a meme prompt for text2image generation (image_prompt) and the top text (top_text) and bottom text if you want to also include the bottom text (bottom_text). The meme will be based on the following: {meme_seed}"
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "email_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "image_prompt": {
                            "description": "meme prompt for text2image generation using Dalle 3",
                            "type": "string"
                        },
                        "top_text": {
                            "description": "top text that will be added to the meme",
                            "type": "string"
                        },
                        "bottom_text": {
                            "description": "bottom optional text that will be added to bottom of the meme",
                            "type": "string"
                        },
                    },
                    "additionalProperties": False,
                    "required": ["image_prompt", "top_text"]
                }
            }
        }
    )

    response_content = response.choices[0].message.content
    parsed_content = json.loads(response_content)
    return parsed_content

def generate_image(image_prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=image_prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    return image_url

def generate_meme(image_url, top_text, bottom_text):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    else:
        image = Image.open(BytesIO(response.content))
        draw = ImageDraw.Draw(image)
        width, height = image.size

        def fit_text(text, max_width, draw_obj, font_obj):
            lines = []
            line = ""
            for word in text.split():
                test_line = f"{line} {word}".strip()
                test_width = draw_obj.textlength(test_line, font=font_obj) if hasattr(draw_obj, 'textlength') else font_obj.getsize(test_line)[0]
                if test_width <= max_width:
                    line = test_line
                else:
                    lines.append(line)
                    line = word
            if line:
                lines.append(line)
            return lines

        max_height = height // 5
        font_size = int(max_height / 2)

        while True:
            font = ImageFont.load_default(font_size)
            top_lines = fit_text(top_text, width - 20, draw, font)
            bottom_lines = fit_text(bottom_text, width - 20, draw, font)
            top_text_height = len(top_lines) * font_size
            bottom_text_height = len(bottom_lines) * font_size
            if top_text_height <= max_height and bottom_text_height <= max_height:
                break
            font_size -= 1

        top_y_position = 20
        bottom_y_position = height - bottom_text_height - 20

        for i, line in enumerate(top_lines):
            draw.text((10, top_y_position + i * font_size), line, font=font, fill="white", stroke_width=2, stroke_fill="black")

        for i, line in enumerate(bottom_lines):
            draw.text((10, bottom_y_position + i * font_size), line, font=font, fill="white", stroke_width=2, stroke_fill="black")

        return image

def process_single_meme(meme_seed):
    meme_data = get_meme(meme_seed)
    image_prompt = meme_data["image_prompt"]
    top_text = meme_data["top_text"]
    bottom_text = meme_data.get("bottom_text", "")
    image_url = generate_image(image_prompt)
    image = generate_meme(image_url, top_text, bottom_text)

    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    image_path = f"data/meme_{timestamp}.png"
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    # image.save(image_path)

    # Log details
    print(f"timestamp: {timestamp}, Meme Seed: {meme_seed}, Image Prompt: {image_prompt}, Top Text: {top_text}, Bottom Text: {bottom_text}, Image Path: {image_path}")

    return image_prompt, top_text, bottom_text, image

def process_multiple_memes(meme_seed):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_single_meme, meme_seed) for _ in range(4)]
        results = [future.result() for future in futures]
    return results

def gradio_app():
    with gr.Blocks() as app:
        gr.Markdown("## Meme Generator")
        with gr.Row():
            with gr.Column(scale=1):
                meme_seed_input = gr.Textbox(label="Meme prompt", value="Wait AI is generating memes now?", placeholder="Enter a meme prompt")
                generate_button = gr.Button("Generate")

            with gr.Column(scale=3):
                with gr.Row():
                    img1 = gr.Image(label="Image 1")
                    img2 = gr.Image(label="Image 2")
                with gr.Row():
                    img3 = gr.Image(label="Image 3")
                    img4 = gr.Image(label="Image 4")
                
        def display_memes(meme_seed):
            results = process_multiple_memes(meme_seed)
            memes_data = []
            for image_prompt, top_text, bottom_text, image in results:
                memes_data.append(image)

            return memes_data

        generate_button.click(
            fn=display_memes,
            inputs=[meme_seed_input],
            outputs=[img1, img2, img3, img4]
        )
    return app

app = gradio_app()
app.launch(debug=True)