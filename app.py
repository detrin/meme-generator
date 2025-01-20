import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from io import BytesIO

import requests
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from openai import OpenAI


class MemeData(BaseModel):
    """
    Pydantic model for meme data returned by the language model.

    Attributes:
        image_prompt (str): A textual prompt to generate an image.
        top_text (str): The top text to place on the generated meme.
        bottom_text (Optional[str]): The bottom text to place on the meme if desired.
    """

    image_prompt: str
    top_text: str
    bottom_text: Optional[str] = None


def get_meme(meme_seed: str, client: OpenAI) -> MemeData:
    """
    Get meme data (image prompt, top text, and optional bottom text)
    based on the provided meme seed.

    Args:
        meme_seed (str): A general idea or backstory for meme generation.
        client (OpenAI): An instance of the OpenAI client (with an API key).

    Returns:
        MemeData: Parsed meme information containing the image prompt,
                  top text, and optional bottom text.
    """

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the best meme lord in the world. You are also highly "
                    "creative, funny and clever."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Generate me a meme prompt for text2image generation "
                    "(image_prompt) and the top text (top_text) and bottom text "
                    "if you want to also include the bottom text (bottom_text). "
                    f"The meme will be based on the following: {meme_seed}"
                ),
            },
        ],
        response_format=MemeData,
    )
    meme = response.choices[0].message.parsed
    return meme


def generate_image(image_prompt: str, client: OpenAI) -> str:
    """
    Generate an image URL based on the provided text prompt using DALL-E 3.

    Args:
        image_prompt (str): Text prompt describing the desired image.
        client (OpenAI): An instance of the OpenAI client (with an API key).

    Returns:
        str: A URL pointing to the generated image.
    """

    response = client.images.generate(
        model="dall-e-3",
        prompt=image_prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response.data[0].url


def generate_meme(
    image_url: str, top_text: str, bottom_text: Optional[str] = None
) -> Image.Image:
    """
    Generate a meme by placing the provided top and bottom text onto the image
    from the given URL.

    Args:
        image_url (str): A URL to the generated image.
        top_text (str): Text to be placed at the top of the image.
        bottom_text (Optional[str]): Text to be placed at the bottom of the image.

    Returns:
        Image.Image: A PIL Image object with the meme text drawn.
    """

    if bottom_text is None:
        bottom_text = ""

    try:
        response = requests.get(image_url)
        response.raise_for_status()
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        raise
    except Exception as err:
        print(f"Other error occurred: {err}")
        raise

    image = Image.open(BytesIO(response.content))
    draw = ImageDraw.Draw(image)
    width, height = image.size

    def fit_text(
        text: str,
        max_width: int,
        draw_obj: ImageDraw.Draw,
        font_obj: ImageFont.ImageFont,
    ) -> list:
        """
        Split text into multiple lines ensuring that each line
        fits within the specified max_width.

        Args:
            text (str): The text to split.
            max_width (int): Maximum allowed width for the text.
            draw_obj (ImageDraw.Draw): PIL drawing object.
            font_obj (ImageFont.ImageFont): Font object for text measurement.

        Returns:
            list: A list of lines that fit within the max_width.
        """
        lines = []
        line = ""
        for word in text.split():
            test_line = f"{line} {word}".strip()
            test_width = (
                draw_obj.textlength(test_line, font=font_obj)
                if hasattr(draw_obj, "textlength")
                else font_obj.getsize(test_line)[0]
            )
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
        draw.text(
            (10, top_y_position + i * font_size),
            line,
            font=font,
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )

    for i, line in enumerate(bottom_lines):
        draw.text(
            (10, bottom_y_position + i * font_size),
            line,
            font=font,
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )

    return image

def process_single_meme(meme_seed: str, client: OpenAI):
    """
    Process a single meme: retrieve meme data, generate the image prompt,
    generate the image from the prompt, and overlay top/bottom text.

    Args:
        meme_seed (str): A general idea or backstory for meme generation.
        client (OpenAI): An instance of the OpenAI client (with an API key).

    Returns:
        tuple: (image_prompt, top_text, bottom_text, PIL.Image object).
    """

    meme_data = get_meme(meme_seed, client)
    image_prompt = meme_data.image_prompt
    top_text = meme_data.top_text
    bottom_text = meme_data.bottom_text

    image_url = generate_image(image_prompt, client)
    image = generate_meme(image_url, top_text, bottom_text)

    # Record details to stdout
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    print(
        f"timestamp: {timestamp}, Meme Seed: {meme_seed}, "
        f"Image Prompt: {image_prompt}, Top Text: {top_text}, "
        f"Bottom Text: {bottom_text}"
    )

    return image_prompt, top_text, bottom_text, image


def process_multiple_memes(meme_seed: str, api_key: str):
    """
    Process multiple memes in parallel using ThreadPoolExecutor.

    Args:
        meme_seed (str): A general idea or backstory for meme generation.
        api_key (str): The OpenAI API key to authenticate requests.

    Returns:
        list: A list of tuples, each containing
              (image_prompt, top_text, bottom_text, PIL.Image object).
    """

    client = OpenAI(api_key=api_key)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_single_meme, meme_seed, client) for _ in range(4)
        ]
        results = [future.result() for future in futures]
    return results


def gradio_app() -> gr.Blocks:
    """
    Build and return the Gradio Blocks application for generating memes.

    Returns:
        gr.Blocks: A Gradio Blocks app object that can be launched.
    """

    with gr.Blocks() as app:
        gr.Markdown("## Meme Generator")

        openai_key_input = gr.Textbox(
            label="OpenAI API Key",
            placeholder="Enter your API key",
            type="password",
        )

        with gr.Row():
            with gr.Column(scale=1):
                meme_seed_input = gr.Textbox(
                    label="Meme prompt",
                    value="Wait AI is generating memes now?",
                    placeholder="Enter a meme prompt",
                )
                generate_button = gr.Button("Generate")

            with gr.Column(scale=3):
                with gr.Row():
                    img1 = gr.Image(label="Image 1")
                    img2 = gr.Image(label="Image 2")
                with gr.Row():
                    img3 = gr.Image(label="Image 3")
                    img4 = gr.Image(label="Image 4")

        def display_memes(meme_seed: str, api_key: str):
            """
            Generate four memes based on the same meme seed and API key,
            and return them for display.

            Args:
                meme_seed (str): A general idea or backstory for meme generation.
                api_key (str): The OpenAI API key for the client.

            Returns:
                list: A list of four PIL.Image objects generated by the process.
            """
            results = process_multiple_memes(meme_seed, api_key)
            memes_data = []
            for _, _, _, image in results:
                memes_data.append(image)
            return memes_data

        generate_button.click(
            fn=display_memes,
            inputs=[meme_seed_input, openai_key_input],
            outputs=[img1, img2, img3, img4],
        )

    return app


if __name__ == "__main__":
    app = gradio_app()
    app.launch(server_name="0.0.0.0", server_port=3000)
