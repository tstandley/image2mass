#import cPickle as pickle
import pickle
import gzip
import lz4
import bisect
import lz4.block

def dump_compressed(to_pickle, filename):
    uncompressed_bytes = pickle.dumps(to_pickle, protocol=4)
    compressed_bytes = lz4.block.compress(uncompressed_bytes)
    with open(filename, 'wb') as fp:
        fp.write(compressed_bytes)

def get_compressed_object(filename):
    with open(filename, 'rb') as fp:
        compressed_bytes = fp.read()
    decompressed = lz4.block.decompress(compressed_bytes)
    pickled_object = pickle.loads(decompressed)

    return pickled_object

def get_gzipped_object(filename):
    with gzip.open(filename, 'rb') as fp:
        uncompressed_bytes = fp.read()
    pickled_object = pickle.loads(uncompressed_bytes)
    return pickled_object

def dump(to_pickle, filename):
    fp = open(filename, 'wb')
    pickle.dump(to_pickle, fp, protocol=4)
    fp.close()

def get_pickled_object(filename):
    #print filename
    with open(filename, 'rb') as fp:
        uncompressed_bytes = fp.read()
    pickled_object = pickle.loads(uncompressed_bytes)
    return pickled_object
    
reduced_density_ranking = get_compressed_object('pklzs/score_to_percentile.pklz')
def score_to_percentile(x,a=None):
    if a is None:
        a = score_to_percentile.a
    mul = 1.0 / float(len(a))
    smaller = bisect.bisect_right(a,x)
    #return smaller * mul
    just_smaller = a[smaller-1]
    if smaller >= len(a):
        return 1.0
    just_bigger = a[smaller]
    if x == just_smaller:
        i = smaller-1
        while a[i] == x:
            i-=1
        i+=1
        i = max(0,i)
        return (i+smaller) * mul*.5
    #print just_smaller, x, just_bigger, smaller
    assert just_smaller <= x and x <= just_bigger, (just_smaller, x, just_bigger, smaller) 
    diff = just_bigger - just_smaller
    amount_more = (x - just_smaller)/diff
    #print amount_more
    return (smaller+amount_more)*mul
score_to_percentile.a = reduced_density_ranking

import sys
def debug(expression):
    frame = sys._getframe(1)

    print(expression, '=', repr(eval(expression, frame.f_globals, frame.f_locals)))


def percentile_to_score(p):
    from numpy import percentile
    return percentile(percentile_to_score.a, p*100)
percentile_to_score.a = reduced_density_ranking


def percentile_to_score_keras(p):
    import numpy as np
    if percentile_to_score_keras.percentile_model is None:
        import keras
        import keras.backend as K
        import tensorflow as tf
        percentile_data = K.constant(reduced_density_ranking,dtype='float32')
        def percentile_to_density(x):
            xx = x*len(reduced_density_ranking)*.99999999
            whole = K.cast(xx, 'int32')
            part = xx-K.cast(whole,'float32')
            lower = tf.gather(percentile_data,whole)
            upper = tf.gather(percentile_data,whole+1)
            return lower*(1.0-part)+part*upper
        pp = keras.layers.Input((1,))
        out = keras.layers.Lambda(percentile_to_density)(pp)
        model = keras.models.Model(inputs=pp,outputs=out)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        percentile_to_score_keras.percentile_model = model
    return percentile_to_score_keras.percentile_model.predict(np.array([p]))[0][0]
percentile_to_score_keras.percentile_model = None

def generator_to_batch_generator(generator, batch_size):
    import traceback
    import numpy as np
    while True:
        if callable(generator):
            gen = generator()
        else:
            gen = generator
        x_accum, y_accum, w_accum = None, None, None
        current_batch_size = 0
        while True:
            try:
                output = next(gen)
            except Exception as e:
                traceback.print_exc()
                if callable(generator):
                    gen = generator()
                continue
            if output is None:
                print('got None from generator')
            else:
                
                if len(output)==2:
                    x,y = output
                    w=None
                if len(output)==3:
                    x,y,w = output
                
                if isinstance(x,tuple) or isinstance(x,list):
                    x_curr = [np.array(xn) for xn in x]
                    if x_accum is None:
                        x_accum = tuple([xx] for xx in x_curr)
                    else:
                        for i,xx in enumerate(x_curr):
                            x_accum[i].append(xx)
                else:
                    if x_accum is None:
                        x_accum = []
                    x_accum.append(np.array(x))
                
                if isinstance(y,tuple) or isinstance(y,list):
                    y_curr = [np.array(yn) for yn in y]
                    if y_accum is None:
                        y_accum = tuple([yy] for yy in y_curr)
                    else:
                        for i,yy in enumerate(y_curr):
                            y_accum[i].append(yy)
                else:
                    if y_accum is None:
                        y_accum = []
                    y_accum.append(np.array(y))
                
                
                if w is not None:
                    if isinstance(w,tuple) or isinstance(w,list):
                        w_curr = [np.array(wn) for wn in w]
                        if w_accum is None:
                            w_accum = tuple([ww] for ww in w_curr)
                        else:
                            for i,ww in enumerate(w_curr):
                                w_accum[i].append(ww)
                    else:
                        if w_accum is None:
                            w_accum = []
                        w_accum.append(np.array(w))
                
                
                current_batch_size+=1
                
                if current_batch_size==batch_size:
                    if isinstance(x_accum,tuple):
                        x_ret = [np.array(xx) for xx in x_accum]
                    else:
                        x_ret = np.array(x_accum)
                    if isinstance(y_accum,tuple):
                        y_ret = [np.array(yy) for yy in y_accum]
                    else:
                        y_ret = np.array(y_accum)
                    if w is not None:
                        if isinstance(w_accum,tuple):
                            w_ret = [np.array(ww) for ww in w_accum]
                        else:
                            w_ret = np.array(w_accum)
                        yield x_ret,y_ret,w_ret
                    else:
                        yield x_ret, y_ret
                        
                    
                    x_accum, y_accum, w_accum = None, None, None
                    current_batch_size = 0
                    

def get_cannonical_transformations():
    import numpy as np
    funcs = [(lambda x: x, lambda x: x),
            (np.fliplr, np.fliplr),
            (np.flipud, np.flipud),
            (lambda x:(np.fliplr(np.flipud(x))), lambda x:(np.flipud(np.fliplr(x)))),
            (np.rot90, lambda x:np.rot90(x, -1)),
            (lambda x: np.flipud(np.rot90(x)), lambda x: np.rot90(np.flipud(x), -1)),
            (lambda x: np.fliplr(np.rot90(x)), lambda x: np.rot90(np.fliplr(x), -1)),
            (lambda x: np.fliplr(np.flipud(np.rot90(x))), lambda x: np.rot90(np.flipud(np.fliplr(x)), -1))]
    return funcs

