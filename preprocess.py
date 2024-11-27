import ccdproc
import os
import numpy as np
import glob
from astropy.nddata import CCDData
import matplotlib.pyplot as plt
from astropy.io import fits
from ccdproc import ImageFileCollection
import astroalign as aa
from astropy import units as u
from scipy import stats

#--------------------------------------------------------------------------------------------
def load_images(path):
    files = glob.glob(os.path.join(path, '*.fits'))  
    if len(files) == 0:
        raise ValueError("해당 경로에 FITS 파일이 없습니다.")
    # ImageFileCollection 객체 생성
    collection = ImageFileCollection(path, glob_include='*.fits')
    return collection

#-------------------------------------------------------------------------------------------
#image_path = 
#zero_path = 
#dark_path = 
#flat_path = 

#master_flat_path = 
#master_zero_path = 
#master_dark_path = 

#-------------------------------------------------------------------------------------------
def master_bias(images):
    
    comment = '-'*60 + '\n' \
              + 'MAKING MASTER ZERO\n' \
              + '-'*60 + '\n'
    print(comment)

    path_data = images.location
    
    zerolist = []
    for hdu, fname in images.hdus(imagetyp='Bias', return_fname=True):
        meta = hdu.header
        zerolist.append(ccdproc.CCDData(data=hdu.data.astype(np.int16), meta=meta, unit="adu")) 
    
    n = 0
    for hdu, fname in images.hdus(imagetyp='Bias', return_fname=True):    
        n += 1
        newmeta = hdu.header.copy() 
        newmeta['FILENAME'] = 'zero.fits'
        newmeta['COMB{}'.format(n)] = fname
    
    print('{} ZERO IMAGES WILL BE COMBINED.'.format(len(zerolist)))

    zeros = ccdproc.Combiner(zerolist)
    mzero = zeros.median_combine()
    mzero.header = newmeta
    mzero.data = mzero.data.astype(np.int16)

    primary_hdu = fits.PrimaryHDU(data=mzero.data, header=mzero.header)

    zero_fits_path = '{}/zero.fits'.format(path_data)
    if zero_fits_path in glob.glob(path_data + '/*'):
        os.remove(zero_fits_path)
    
    hdul = fits.HDUList([primary_hdu])
    hdul.writeto(zero_fits_path, overwrite=True)

    return mzero

#---------------------------------------------------------------------------------------
def master_dark(images, master_zero_path):
    # 로그 출력
    comment = '-'*60 + '\n' \
              + 'MAKING MASTER DARK\n' \
              + '-'*60 + '\n'
    print(comment)

    path_data = images.location
    mdark_name = 'dark.fits'

    with fits.open(master_zero_path) as hdul: 
        mzero_data = hdul[0].data.astype(np.int16)
        mzero_header = hdul[0].header
        mzero = CCDData(data=mzero_data, meta=mzero_header, unit="adu")

    zdarklist = []
    for hdu, fname in images.hdus(imagetyp='dark', return_fname=True):
        meta = hdu.header
        dark = CCDData(data=hdu.data.astype(np.int16), meta=meta, unit="adu")
        zdark = ccdproc.subtract_bias(dark, mzero)
        zdark.meta['SUBBIAS'] = mzero.meta['FILENAME']
        zdarklist.append(ccdproc.CCDData(data=zdark.data.astype(np.int16), meta=meta, unit="adu"))

    newmeta = None
    n = 0
    for hdu, fname in images.hdus(imagetyp='dark', return_fname=True):    
        n += 1
        newmeta = hdu.header.copy()
        newmeta['FILENAME'] = 'master_dark.fits'
        newmeta['COMB{}'.format(n)] = fname

    print('{} DARK IMAGES WILL BE COMBINED.'.format(len(zdarklist)))

    darks = ccdproc.Combiner(zdarklist)
    mdark = darks.median_combine()
    mdark.header = newmeta
    mdark.data = mdark.data.astype(np.int16)

    primary_hdu = fits.PrimaryHDU(data=mdark.data, header=mdark.header)

    dark_file_path = os.path.join(path_data, mdark_name)
    if os.path.exists(dark_file_path):
        os.remove(dark_file_path)

    hdul = fits.HDUList([primary_hdu])
    hdul.writeto(dark_file_path, overwrite=True)

    return mdark

#------------------------------------------------------------------------------------
def master_flat(images, master_zero_path, master_dark_path, filte):
    comment = '-' * 60 + '\n' \
              + 'MAKING MASTER FLAT\n' \
              + '-' * 60 + '\n'
    print(comment)

    with fits.open(master_zero_path) as hdul:
        mzero_data = hdul[0].data.astype(np.int16)
        mzero_header = hdul[0].header
        mzero = CCDData(data=mzero_data, meta=mzero_header, unit="adu")

    with fits.open(master_dark_path) as hdul:
        mdark_data = hdul[0].data.astype(np.int16)
        mdark_header = hdul[0].header
        mdark = CCDData(data=mdark_data, meta=mdark_header, unit="adu")

    print('SUBTRACT MASTER ZERO & DARK FROM FLAT FRAMES')
    zflatlist = []

    for hdu, fname in images.hdus(return_fname=True):
        zflat = ccdproc.subtract_bias(flat, mzero)

        dzflat = ccdproc.subtract_dark(
                ccd=zflat, master=mdark,
                exposure_time='EXPTIME',
                exposure_unit=u.second,
                scale=False
            )

        meta['SUBBIAS'] = mzero.meta['FILENAME']
        meta['SUBDARK'] = mdark.meta['FILENAME']
        zflatlist.append(dzflat)

    print(len(zflatlist))
    outname = 'n' + filte + '.fits'
    newmeta = meta
    newmeta['FILENAME'] = outname
    n = 0
    
    for hdu, fname in images.hdus(imagetyp='flat', filter=filte, return_fname=True):
        n += 1
        newmeta['COMB{}'.format(n)] = fname
    
    print('{} FLAT IMAGES WILL BE COMBINED.'.format(len(zflatlist)))
    flat_combiner = ccdproc.Combiner(zflatlist)
    flat_combiner.minmax_clipping()

    def scaling_func(arr): return 1 / np.ma.median(arr)
    flat_combiner.scaling = scaling_func

    mflat = flat_combiner.median_combine()
    mflat.header = newmeta
    mflat.data = mflat.data.astype(np.int16)

    primary_hdu = fits.PrimaryHDU(data=mflat.data, header=mflat.header)

    path_data = images.location 
    flat_file_path = os.path.join(path_data, outname)
    
    if os.path.exists(flat_file_path):
        os.remove(flat_file_path)
        
    hdul = fits.HDUList([primary_hdu])
    hdul.writeto(flat_file_path, overwrite=True)

    return mflat

#------------------------------------------------------------------------------------------
def calibration(images, mzero, mdark, mflat, filte):
	comment     = '-'*60+'\n' \
				+ 'OBJECT CORRECTION\n' \
				+ '-'*60+'\n'
	print(comment)
	path_data = images.location
	objlist = []
	for hdu, fname in images.hdus(imagetyp='object', filter=filte, return_fname=True):
		meta = hdu.header
		objlist.append(meta['object'])
		obj = ccdproc.CCDData(data=hdu.data, meta=meta, unit="adu")
		
		zobj = ccdproc.subtract_bias(obj, mzero)
		meta['SUBBIAS'] = mzero.meta['FILENAME']
		
		zobj = ccdproc.subtract_dark(ccd=zobj, master=mdark,
							   exposure_tiem='EXPTIME',
							   exposure_unit=u.second,
							   scale=True)
		meta['SUBDARK'] = mdark.meta['FILENAME']
		
		fzobj = ccdproc.flat_correct(zobj, mflat)
		meta['DIVFLAT'] = mflat.meta['FILENAME']
		
		fzobj.header = meta
		fzobj.data = fzobj.data.astype(np.int16)
		primary_hdu = fits.PrimaryHDU(data=fzobj.data, header=fzobj.header)
		outname = images + filte + '.fits'
		im_path = os.path.join(path_data, outname)
		hdul = fits.HDUList([primary_hdu])
		hdul.writeto(im_path, overwirte=True)
#--------------------------------------------------------------------------------------------

    