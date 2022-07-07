import Adafruit_GPIO as GPIO
import Adafruit_SSD1306
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import subprocess
import time

disp = Adafruit_SSD1306.SSD1306_128_32(rst=None, i2c_bus=0, gpio=1)
disp.begin()

disp.clear()
disp.display()
disp_image = Image.new('1', (disp.width, disp.height))

screen = ImageDraw.Draw(disp_image)
screen.rectangle((0, 0, disp.width, disp.height), outline=0, fill=0)
screen_padding= -2
screen_top = screen_padding
screen_bottom = disp.height - screen_padding
screen_x = 0

font = ImageFont.load_default()


def print_led(text1 = '', text2='', text3='', text4=''):
    screen.rectangle((0, 0, disp.width, disp.height), outline=0, fill=0)

    # if you want to print a message on the top line.
    screen.text((screen_x, screen_top+0), text1, font=font, fill=255)
    screen.text((screen_x, screen_top+8), text2, font=font, fill=255)
    screen.text((screen_x, screen_top+16), text3, font=font, fill=255)
    screen.text((screen_x, screen_top+24), text4, font=font, fill=255)
    
    disp.image(disp_image)
    disp.display()
    

