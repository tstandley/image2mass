import sys
from PIL import Image
import numpy as np

#import image_util
from model_wrapper import ComplexModel


shape_aware_model = ComplexModel()

def predict_mass(filename,dims):
    global shape_aware_model
    im = Image.open(filename)
    #im = image_util.resize_and_pad_image(im,(299,299))
    im = np.array(im)

    output = shape_aware_model.predict((im,dims))
    return output

def main():
    try:
        filename, l,w,h = sys.argv[1:]
    except ValueError:
        print("Usage: $ python3 predict_mass.py filename length width height")
        print("Length, width, and height must be floating point numbers in inches")
        print("Example: $ python3 predict_mass.py filename length width height")
        sys.exit(1)


    print("Got: filename=",filename,'dimensions= (', l, 'inches by', w, 'inches by', h, 'inches.)')
    
    dims = (float(l),float(w),float(h))
    output = predict_mass(filename,dims)
    print(filename, 'probably weighs about', output, 'pounds.')

if __name__ == "__main__":
    main()