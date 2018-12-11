# image2mass

Code for http://proceedings.mlr.press/v78/standley17a.html

## Run in Google Colab
1. Go to https://colab.research.google.com/
2. File > Upload notebook... > GitHub
3. Paste https://github.com/Dene33/image2mass/blob/master/image2mass_colab.ipynb
4. Select `image2mass_colab.ipynb`

or just go to: https://github.com/Dene33/image2mass/blob/master/image2mass_colab.ipynb and click `Open in Colab`

## setup (works only with python3):
Install tensorflow

Download evaluation datafiles (see below)

### use pip
```
pip install Pillow
pip install keras==2.1.1 # the most recent version (2.1.2 at time of writing) throws an error loading the model
pip install lz4
pip install opencv-python # or install opencv another way (this method doesn't support cv.imshow() or FFmpeg) but is much easier to install
pip install h5py
```
### conda
Coming soon


## example usage:
```
python3 predict_mass.py test_set_images/airplane_clock_1.jpg 6.25 1.25 2.125 2>/dev/null
```
(more in predictions.txt)

## Ground Truth for Household Test Set:

https://docs.google.com/spreadsheets/d/1-cjqxaG8AGP14KvUZxSQzLGfGmS3_IbVO47_vbBKyoc

## Evaluation Data Files:

https://drive.google.com/drive/folders/17yEukxIyjyen3vJ8nuX_YklnsVBZBol4

## Amazon train/val/test Data Files

https://goo.gl/forms/hCLNHOcv7NRkEQP03
