from PIL import Image, ImageDraw, ImageFont
import numpy as np

class EmojiDisplay:
    def __init__(self):
        self.background_color = (0, 0, 0, 0)
        self.font_color = (255, 255, 255, 255)
        self.font = ImageFont.truetype('AppleColorEmoji.ttf',137)
        self.colour_mode = "RGBA"
        self.size = (200, 200)
    
    def make_image_scratch(self, emoji_text):
        image = Image.new(self.colour_mode, self.size, self.background_color)
        return self.draw_emoji(image, emoji_text)
    
    def draw_emoji(self, image, emoji_text):
        # unicode_text = emoji_text.encode('utf-8') # not needed?
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), emoji_text, font=self.font, embedded_color=True)
        return image

    def compose_images(self, image, emoji_text):
        # convert opencv image to PIL form
        image_pil = Image.fromarray(image)
        image_pil = self.draw_emoji(image_pil, emoji_text)
        # convert back to np/opencv
        return np.array(image_pil)
