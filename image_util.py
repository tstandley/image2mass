
from PIL import Image
def resize_image(image, desired_size):
    scale_factor = min(float(desired_size[0])/image.size[0],float(desired_size[1])/image.size[1])
    scaled_size = (int(image.size[0]*scale_factor+.0001), int(image.size[1]*scale_factor+.0001))
    #print scale_factor, scaled_size,
    scaled_image = image.resize(scaled_size, Image.ANTIALIAS)
    return scaled_image

def pad_image(image, padded_size):
    new_image = Image.new('RGB', padded_size, (255,255,255))
    scaled_size = image.size
    new_image.paste(image,((padded_size[0]-scaled_size[0])//2,(padded_size[1]-scaled_size[1])//2))
    return new_image

def pad_image_bottom_left(image, padded_size):
    new_image = Image.new('RGB', padded_size, (255,255,255))
    scaled_size = image.size
    new_image.paste(image,(0,padded_size[1]-scaled_size[1]))
    return new_image
    
def resize_and_pad_image(image, desired_size):
    scaled_image = resize_image(image,desired_size)
    padded_image = pad_image(scaled_image,desired_size)
    return padded_image

def getImageFromString(encoded_image):
    buff = io.BytesIO() #buffer where image is stored
    buff.write(encoded_image)
    buff.seek(0)
    img = Image.open(buff)
    return img

def encodeImage(image):
    buff = io.BytesIO()
    #image.save(buff, format='WebP', lossless=True)
    #image.save(buff, format='jpeg')
    image.save(buff, format='WebP', quality=88)
    buff.seek(0)
    str= buff.read()
    return str

def showImage(encoded_image):
    buff = io.BytesIO() #buffer where image is stored
    buff.write(encoded_image)
    buff.seek(0)
    img = Image.open(buff)
    img.show()