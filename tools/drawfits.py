import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
# from matplotlib.colors import LogNorm
cnt_file = 46
cnt_img = 100
dis_data = np.zeros((cnt_file, cnt_img, cnt_img), dtype=float)

fromstart = np.zeros(cnt_file, dtype=float)
fits_files = []
fits_times = []

fig, axes = plt.subplots(1, 1)# gridspec_kw={'width_ratios':[10, 1]})

DIR = "E://Data//SGR//121//cut//"

for root, _, files in os.walk(DIR):
        for file in files:
            if os.path.splitext(file)[1] == '.fits':
                fits_files.append(os.path.join(root, file))

fits_files.sort()
# for i in range(len(fits_files)):
#     print(os.path.basename(fits_files[i]).split('_')[0])
# exit()
for s in range(cnt_file):
    fits_times.append(os.path.basename(fits_files[s]).split('_')[0])
    if s > 0:
        fromstart[s] =  (int(fits_times[s])-int(fits_times[0]))*10//60/10
    hdu_list = fits.open(fits_files[s], memmap=True)
    # hdu_list.info()
    evt_data = Table(hdu_list[0].data)
    for ss in range(cnt_img):
        dis_data[s, ss] = evt_data.columns[ss+50][50:150]
# print(dis_data.shape)
# while(1):
for s in range(cnt_file):
    # # img_zero_mpl = plt.imshow(dis_data, cmap='viridis', norm=LogNorm())
    # fig, axes = plt.subplots(1, 1)
    img_zero_mpl = axes.imshow(np.transpose(dis_data[s]), 
                aspect = 'auto', origin = 'lower',
                vmin = dis_data[s].mean()-1, vmax = dis_data[s].mean()+1,
                cmap = 'plasma')  # viridis, magma, Blues
    pst1 = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cb1 = fig.colorbar(img_zero_mpl, cax=pst1)
    # cbar.ax.set_yticklabels(['1','3','6'])
    plt.suptitle('Number:' + str(s+1) + '  Time:' + fits_times[s] +
        '  FromStart:' + str(fromstart[s])+'min')
    # plt.savefig('E://Data//SGR//121//png//test.%02d.png'%(s+1))
    # plt.close()
    plt.pause(0.1) 
    # plt.pause(2.0)
# plt.show()
