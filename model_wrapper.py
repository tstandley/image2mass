import keras
import numpy as np
import os

import util
import features
import cv2
import numba


def get_model(model_contains):
    newest = max(map(lambda x: 'models/'+x, [x for x in os.listdir('models') if '.h5' in x and model_contains in x]), key=os.path.getctime)
    print(newest)
    model = keras.models.load_model(newest)
    return model, newest[7:]

def get_points_from_mask(mask):

    cntr = []
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            t = mask[x][y]
            if t > .001:
                cntr.append([[y,x]])
    return cntr

def get_rect_from_mask(mask):
    cntr=get_points_from_mask(mask)
    if len(cntr)<3:
        return [(16,16),[0,0],0]
    try:
        rect = cv2.minAreaRect(np.array(cntr))
    except:
        return [(16,16),[0,0],0]
    return rect

def predict(model, image):
    cannonical_transformations = util.get_cannonical_transformations()
    
    xes = []
    for func, inverse in cannonical_transformations:
        xes.append(func(image))
    
    ys = model.predict(np.array(xes))
    y = np.zeros_like(ys[0])
    for i,(func, inverse) in enumerate(cannonical_transformations):
        better = inverse(ys[i])
        y+=better
        
    y /= len(cannonical_transformations)
    return y

@numba.jit
def get_filter_mask(img,threshold=250):
    mask = (np.sum(img,axis=2) < threshold*3).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)).astype(np.uint8)
    
    ret = mask
    ret = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)).astype(np.uint8)
    ret = cv2.erode(ret,kernel,iterations = 1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4)).astype(np.uint8)
    ret = cv2.morphologyEx(ret, cv2.MORPH_OPEN, kernel)
    
    ret = ret.astype(np.bool)
    return ret

def get_mask_and_rect(thickness_model,img):
    small_img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
    small_img = np.asarray(small_img)
    mask = predict(thickness_model,small_img)
    binary_mask = get_filter_mask(small_img)
    coarse_binary_mask = cv2.resize(binary_mask.astype(np.uint8),(32,32), interpolation=cv2.INTER_AREA) > .001
        
    mask = mask * coarse_binary_mask
    rect = get_rect_from_mask(mask)
    ((rx,ry),(rw,rh),rtheta) = rect
    rect = np.array([rx,ry,rw,rh,rtheta])
    mask = cv2.resize(mask, (19,19), interpolation=cv2.INTER_AREA)
    return mask,rect

def dummy_loss(a,b):
    return a

class ComplexModel:
    def __init__(self,simple=False):
        if simple == 'no_geometry':
            self.model, self.model_name = get_model('no_geometry')
        elif not simple:
            self.model, self.model_name = get_model('shape_aware')
        else:
            self.model, self.model_name = get_model('cnn_w_alde')
        self.input_shape=(None,299,299,3)

        self.thickenss_model = None
        
    def predict(self, x):
        if len(x) ==2:
            img, dims = x
            if self.thickenss_model is None:
                self.thickenss_model = keras.models.load_model('thickness_model.h5',{'VMAP':dummy_loss,'IoU2d':dummy_loss,'VMAP':dummy_loss, 'VR2':dummy_loss})
            
            mask, rect = get_mask_and_rect(self.thickenss_model,img)
            if img.shape != (299,299,3):
                img=cv2.resize(img, (299,299), interpolation=cv2.INTER_AREA)
        else:
            img,mask,dims,rect = x

        dummy=0
        item = (img,mask,dummy,dims,rect)
        
        img,mask,aux_input = features.prepare_features(item)

                    
        cannonical_transformations = util.get_cannonical_transformations()

        imgs = []
        masks = []
        auxes = []
        for func, _ in cannonical_transformations:
            imgs.append(func(img))
            masks.append(func(mask))
            auxes.append(np.copy(np.array(aux_input)))
        predictions = self.model.predict([np.array(imgs),np.array(masks),np.array(auxes)])
            
        return np.median(predictions)
    
class SimpleModel:
    def __init__(self):
        self.model, self.model_name = get_model('pure_cnn')
        self.input_shape=(None,299,299,3)
    def predict(self, x):
        img,mask,dims,rect = x

        cannonical_transformations = util.get_cannonical_transformations()
        
        imgs = []
        for func, _ in cannonical_transformations:
            imgs.append(func(img))
        predictions = self.model.predict(np.array(imgs))
            
        return util.percentile_to_score(np.median(predictions))*np.prod(dims)/27.6799