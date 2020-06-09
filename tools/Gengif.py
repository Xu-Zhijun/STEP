import matplotlib.pyplot as plt
import imageio,os

TIME_GAP = 1.0
# DIR = "E://Data//SGR//121//png//"
DIR = "D://ASKAP-PNG//"
# gif_name = 'E://Data//SGR//121//test.gif'
gif_name = 'D://askap.gif'

png_files = []
frames = []
s = 0

for root, _, files in os.walk(DIR):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                png_files.append(os.path.join(root, file))
png_files.sort()
for image_name in png_files:
    s += 1
    # print(s, len(png_files), image_name)
    frames.append(imageio.imread(image_name))
imageio.mimsave(gif_name, frames, 'GIF', duration = TIME_GAP)