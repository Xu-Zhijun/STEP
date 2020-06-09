from astropy.io import ascii
import numpy as np
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt

data = ascii.read("frbcat.csv", encoding='utf-8')

ra = coord.Angle(data['rop_raj'], unit=u.hour)
ra = ra.wrap_at(180*u.degree)
dec = coord.Angle(data['rop_decj'], unit=u.degree)

dec_pushchino = dec[np.where(data['telescope'] == 'Pushchino')]
ra_pushchino = ra[np.where(data['telescope'] == 'Pushchino')]

dec_chime = dec[np.where(data['telescope'] == 'CHIME/FRB')]
ra_chime = ra[np.where(data['telescope'] == 'CHIME/FRB')]

dec_askap = dec[np.where(data['telescope'] == 'ASKAP')]
ra_askap = ra[np.where(data['telescope'] == 'ASKAP')]

dec_parkes = dec[np.where(data['telescope'] == 'parkes')]
ra_parkes = ra[np.where(data['telescope'] == 'parkes')]

dec_utmost = dec[np.where(data['telescope'] == 'UTMOST')]
ra_utmost = ra[np.where(data['telescope'] == 'UTMOST')]

dec_out = dec[np.where(dec > 30*u.degree)]
ra_out = ra[np.where(dec > 30*u.degree)]

frb_out = data['frb_name'][np.where(dec > 30*u.degree)]
print('FRBs Dec larger than 30: ')
for i in frb_out: print(i, end='\n')

with plt.rc_context({'axes.edgecolor':'w', 'xtick.color':'w', 'ytick.color':'w', 
    'figure.facecolor':'k'}):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.set_facecolor('dimgray')
    # ax.invert_xaxis()

    p1 = ax.scatter(ra.radian, dec.radian, s= 3, c='#0000FF')                       # 'blue'
    # ax.scatter(ra_out.radian, dec_out.radian, s= 3, c='#A9A9A9')                  # 'darkgray'
    p2 = ax.scatter(ra_pushchino.radian, dec_pushchino.radian, s= 3, c='#FFA500')   # 'orange'
    p3 = ax.scatter(ra_chime.radian, dec_chime.radian, s= 3, c='#FF0000')           # 'red'
    p4 = ax.scatter(ra_parkes.radian, dec_parkes.radian, s= 3, c='#FFFF00')         # 'yellow'
    p5 = ax.scatter(ra_utmost.radian, dec_utmost.radian, s= 3, c='#00FFFF')         # 'cyan'
    p6 = ax.scatter(ra_askap.radian, dec_askap.radian, s= 3, c='#00FF00')           # 'lime'
    
    ax.set_xticklabels(['14h','16h','18h','20h','22h','0h','2h','4h','6h','8h','10h'])
    ax.grid(True)
    plt.grid(linestyle='--', c='gray')

    plt.legend([p6, p3, p4, p5, p2, p1], ['ASKAP ' + str(len(ra_askap)), 
        'CHIME/FRB ' + str(len(ra_chime)), 'Parkes ' + str(len(ra_parkes)), 
        'UTMOST ' + str(len(ra_utmost)), 'Pushchino ' + str(len(ra_pushchino)),
        'Others ' + str(len(ra)- len(ra_chime) - len(ra_askap) - len(ra_utmost) -
        len(ra_parkes) - len(ra_pushchino))], 
        loc='best')
    plt.savefig('skymap.png',facecolor='k')
# plt.show()