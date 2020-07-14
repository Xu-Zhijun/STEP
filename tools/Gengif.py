import matplotlib.pyplot as plt
import imageio
import os

TIME_GAP = 0.5
Maxnum = 20
# DIR = "E://Data//SGR//121//png//"
# DIR = "D://ASKAP-PNG//"
DIR = "D://My_Designs//Python//FScorr//png//png"
DIR = "D://文档//工作文档//2020//SKA//SGR//MWA TEL"
DIR = 'D:/My_Designs/Python/STEP/ASKAP/FRB170906/png/frb'
# gif_name = 'E://Data//SGR//121//test.gif'
# gif_name = 'D://askap.gif'
gif_name = 'D://frb.gif'

png_files = []
frames = []
s = 0

for root, _, files in os.walk(DIR):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                png_files.append(os.path.join(root, file))
png_files = png_files[ : Maxnum]
png_files.sort()
for image_name in png_files:
    s += 1
    # print(s, len(png_files), image_name)
    frames.append(imageio.imread(image_name))
imageio.mimsave(gif_name, frames, 'GIF', duration = TIME_GAP)