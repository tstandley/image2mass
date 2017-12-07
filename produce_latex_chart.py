#!/usr/bin/env python3

import keras
import time

from keras.preprocessing import image as keras_image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import sys
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as preprocess_input_xception
import progressbar
import util
import tensorflow as tf
from math import log
from model_wrapper import ComplexModel, SimpleModel



def get_amazon_test_set():
    data = util.get_compressed_object('pklzs/amazon_test_set.pklz')
    return data

def get_household_test_set():
    ret, ret2 = util.get_compressed_object('pklzs/household_test_set.pklz')
    return (ret,ret2)

def test_model(model,dataset_real=False,group_images=False):
    model_name = model.model_name
    if dataset_real:
        print()
        print()
        print ('################# Testing using real data with model, '+model_name+" #####################")
        print()
        print()
        a_test_set, binned_test_set = get_household_test_set()
        if group_images:
            a_test_set = binned_test_set.values()
            
    else:
        print()
        print()
        print ('################ Testing using amazon data with model, '+model_name+" ####################")
        print()
        print()
        a_test_set = get_amazon_test_set()
        
    print()

    print('Found', len(a_test_set), 'test set examples...')
    
    reg_y_preds = []
    reg_y_trues= []
    log_y_preds = []
    log_y_trues= []
    hard_y_preds = []
    hard_y_trues= []

    bar = progressbar.ProgressBar()
    
    mape_error = []
    max_error = []
    
    min_error = []
    loss_error = []
    mean_gt = []
    mean_less_2_factor = []
    errors_to_values = []
    t0 = time.time()
    for item in bar(a_test_set):
        if not group_images:
            img, mask, density, dims, rect, weight = item
            
            x = img,mask,dims,rect
            y_true = weight
            y_pred = model.predict(x)
        else:
            y_pred = []
            for item2,image_name in item:
                img, mask, density, dims, rect, weight = item2
                x = img,mask,dims,rect
                y_true = weight
                pred = model.predict(x)
                y_pred.append(pred)
                
            y_pred = np.median(y_pred)
            
        taken = time.time()-t0
        ape = abs(y_pred-y_true)/y_true
        mape_error.append(ape)
        mxe = max(y_pred/y_true,y_true/y_pred)
        max_error.append(mxe)
        mne=1.0/mxe
        min_error.append(mne)
        mean_less_2_factor.append(mxe<2)
        loss_error.append(abs(log(y_pred+.000001)-log(y_true+.000001)))
        mean_gt.append(y_pred>y_true)
        
        hard_y_trues.append(y_true/np.prod(dims))
        hard_y_preds.append(y_pred/np.prod(dims))
        reg_y_trues.append(y_true)
        reg_y_preds.append(y_pred)    
        log_y_trues.append(np.log(y_true))
        log_y_preds.append(np.log(y_pred))
        errors_to_values.append((mxe,mne,ape,y_pred,y_true, item))
    
    print('Took ', taken,' seconds to predict ', len(a_test_set),' items.', taken/ len(a_test_set), ' seconds per item.')
    print('predicted is larger', np.mean(mean_gt), 'of the time')
    
    print('mape_error =', np.mean(mape_error))
    print('max_error =', np.mean(max_error))
    print('min_error =\033[1m', np.mean(min_error),'\033[0m')
    print('loss_error =', np.mean(loss_error))
    print('worse than factor of 2 =', np.mean(mean_less_2_factor))
    
    def e(l): #computes the mean and standard deviation and returns the tuple
        ret = (np.mean(l),np.std(l,ddof=1)/np.sqrt(len(l)))
        #print(len(l),ret)
        return ret
    
    y_trues = log_y_trues
    y_preds = log_y_preds

    z = np.polyfit(y_preds, y_trues, 1)
    print('final z=',z)
    p = np.poly1d(z)
    print('final p=',p)
    correlation = np.corrcoef(y_trues, y_preds)[0,1]
    print('correlation coefficient=', correlation, 'r^2=', correlation**2)
    
    errors = [e(loss_error),
              e(mape_error),
              correlation**2,
              #e(max_error),
              e(min_error),
              e(mean_less_2_factor)]
    
    sys.stdout.flush()
    return errors

def keras_knn(xes, y1es, y2es, k=5):
    inp = keras.Input(shape=(xes.shape[1],), dtype='float32')
    x_data = keras.backend.variable(xes.T, dtype='float32')
    y1_data = keras.backend.variable(y1es, dtype='float32')
    y2_data = keras.backend.variable(y2es, dtype='float32')
    
    def do_dot(x):
        return keras.backend.dot(x, x_data)
    b = keras.layers.Lambda(do_dot)(inp)

    def get_top_key_y1(x):
        values, indices = tf.nn.top_k(x, k, sorted=False)
        return tf.gather(y1_data, indices)
    def get_top_key_y2(x):
        values, indices = tf.nn.top_k(x, k, sorted=False)
        return tf.gather(y2_data, indices)
    b1 = keras.layers.Lambda(get_top_key_y1)(b)
    b1 = keras.layers.Lambda(lambda x:keras.backend.mean(x, axis=-1))(b1)
    b2 = keras.layers.Lambda(get_top_key_y2)(b)
    b2 = keras.layers.Lambda(lambda x:keras.backend.mean(x, axis=-1))(b2)
    model = keras.models.Model(inputs=inp, outputs=[b1, b2])
    #dummy stuff to give to compile because it's required
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    return model

def prepare_nearest_neighbor():
    class KnnModel:
        def __init__(self,xes,y1es,y2es, image_net_model_type = 'vgg',k=40, weight=False):
            print('k =',k, 'model_type= ',image_net_model_type)
            print('setting up model...')
            self.keras_model = keras_knn(xes,y1es,y2es, k=k)
            print('...done')
            
            self.model_name='knn_'+image_net_model_type
            self.weight=weight
            if weight:
                self.model_name=self.model_name+'_weight'
            self.image_net_model_type = image_net_model_type
            self.xes = xes.T
            self.y1es = y1es
            self.y2es = y2es
            
            if image_net_model_type=='vgg':
                base_vgg_model = VGG16(weights='imagenet')
                self.image_net_model = keras.models.Model(inputs=base_vgg_model.input, outputs=base_vgg_model.get_layer('fc1').output)
            elif image_net_model_type == 'resnet':
                self.image_net_model = ResNet50(include_top=False, weights='imagenet')
            elif image_net_model_type == 'xception':
                xception_base_model = Xception(include_top=True, weights='imagenet')
                self.image_net_model = keras.models.Model(inputs=xception_base_model.input, outputs=xception_base_model.get_layer('avg_pool').output)
            self.input_shape = self.image_net_model.input_shape
            print('Finished constructor')

        def predict(self,x):
            img,mask,dims,rect = x
            if len(img.shape)!=3:
                ret = []
                for im in img:
                    ret.append(self.predict(im))
                return np.array(ret)
            new_im = keras_image.img_to_array(img)
            new_im = np.expand_dims(new_im, axis=0)
            if self.image_net_model_type in ['vgg','resnet']:
                preprocessed_im = preprocess_input(new_im)
            elif self.image_net_model_type == 'xception':
                preprocessed_im = preprocess_input_xception(new_im)
            
            features = self.image_net_model.predict(preprocessed_im)
            features = np.squeeze(features)
            prediction = self.keras_model.predict(np.array([features]))
            if self.weight:
                return prediction[0][0]
            return prediction[1][0]*np.prod(dims)/27.6799

            
    #xes_vgg, yes = util.get_compressed_object('vgg_nearest_neighbor_data.pkl')
    #knn_model = KnnModel(xes_vgg,yes,image_net_model_type='vgg')
    #xes_resnet, yes = util.get_compressed_object('resnet_nearest_neighbor_data.pkl')
    #knn_model = KnnModel(xes_resnet,yes,image_net_model_type='resnet')
    xes_xception, y1es, y2es = util.get_pickled_object('pklzs/xception_nearest_neighbor_data.pkl')
    knn_model = KnnModel(xes_xception,y1es,y2es,image_net_model_type='xception')
    return knn_model
    

if __name__=='__main__':
    
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 2/4
    set_session(tf.Session(config=config))
    shape_aware = ComplexModel()
    pure_cnn = SimpleModel()
    
    
    c_errors_r = test_model(model=shape_aware,dataset_real=True)
    s_errors_r = test_model(model=pure_cnn,dataset_real=True)
    c_errors_f = test_model(model=shape_aware,dataset_real=False)
    s_errors_f = test_model(model=pure_cnn,dataset_real=False)
    c_errors_rb = test_model(model=shape_aware,dataset_real=True,group_images=True)
    keras.backend.clear_session()
    cnn_w_alde = ComplexModel(simple=True)
    no_geometry = ComplexModel(simple='no_geometry')
    v_errors_r = test_model(model=cnn_w_alde,dataset_real=True)
    v_errors_f = test_model(model=cnn_w_alde,dataset_real=False)
    a_errors_r = test_model(model=no_geometry,dataset_real=True)
    a_errors_f = test_model(model=no_geometry,dataset_real=False)
    keras.backend.clear_session()
    knn_model = prepare_nearest_neighbor()
    k_errors_r = test_model(model=knn_model,dataset_real=True)
    k_errors_f = test_model(model=knn_model,dataset_real=False)
    knn_model.weight = True
    knn_model.model_name += '_weight'
    w_errors_r = test_model(model=knn_model,dataset_real=True)
    w_errors_f = test_model(model=knn_model,dataset_real=False)
         
    
    def print_errors_for_latex(model_name,errors):
        print(model_name+'&',end='')
        for i,v in enumerate(errors):
            if isinstance(v, str):
                print(v, end='')
            else:
                try:
                    m,sd = v
                    print('{:04.3f}\\pm{:04.3f}'.format(m,sd*2), end='')
                except:
                    m=v
                    print('{:04.3f}'.format(m), end='')
            if i!=len(errors)-1:
                print('&',end='')
        print('\\\\')
     
    print('\t'+r'\small model&\small mALDE \left(\downarrow\right)&\small mAPE \left(\downarrow\right)&\small $r^2_{ls}$ \left(\uparrow\right)&\small \textbf{mMnRE} \left(\uparrow\right)&\small $q$ \left(\uparrow\right)\\')
    print('\t\\midrule')
    print_errors_for_latex('\tXception \\textit{k}-NN',w_errors_r)
    print_errors_for_latex('\tXception \\textit{k}-NN (SIM)',k_errors_r)
    print_errors_for_latex('\tPure CNN',s_errors_r)
    print_errors_for_latex('\tCNN with ALDE',v_errors_r)
    print_errors_for_latex('\tNo Geometry',a_errors_r)
    print_errors_for_latex('\tShape-aware',c_errors_r)
    print('\t\\midrule')
    print_errors_for_latex('\tShape-aware multi-view',c_errors_rb)
    print('\t\\midrule')

    print_errors_for_latex('\tHuman Median',[(.45887,.06578),(.58324,.26111),.628,(.675088,.03123),(.79824,.084293)])
    print_errors_for_latex('\tHuman Ensemble',[(.322614,.040983),(.398139,.072900),.804,(.753706,.025153),(.877192,.044249)])
    print()
    print()
    print('\t'+r'\small model&\small mALDE \left(\downarrow\right)&\small mAPE \left(\downarrow\right)&\small $r^2_{ls}$ \left(\uparrow\right)&\small \textbf{mMnRE} \left(\uparrow\right)&\small $q$ \left(\uparrow\right)\\')
    print('\t\\midrule')
    print_errors_for_latex('\tXception \\textit{k}-NN',w_errors_f)
    print_errors_for_latex('\tXception \\textit{k}-NN (SIM)',k_errors_f)
    print_errors_for_latex('\tPure CNN',s_errors_f)
    print_errors_for_latex('\tCNN with ALDE',v_errors_f)
    print_errors_for_latex('\tNo Geometry',a_errors_f)
    print_errors_for_latex('\tShape-aware',c_errors_f)
