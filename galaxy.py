#!/usr/bin/python

import sys, os
import cv2
import numpy as np
import scipy.ndimage as nd
import csv
import math

from scipy.stats.mstats import mquantiles, kurtosis, skew

def get_data_folder():
    return "model_folder"

def get_fieldnames():
    return [ 'GalaxyID',
             'Class1.1',
             'Class1.2',
             'Class1.3',
             'Class2.1',
             'Class2.2',
             'Class3.1',
             'Class3.2',
             'Class4.1',
             'Class4.2',
             'Class5.1',
             'Class5.2',
             'Class5.3',
             'Class5.4',
             'Class6.1',
             'Class6.2',
             'Class7.1',
             'Class7.2',
             'Class7.3',
             'Class8.1',
             'Class8.2',
             'Class8.3',
             'Class8.4',
             'Class8.5',
             'Class8.6',
             'Class8.7',
             'Class9.1',
             'Class9.2',
             'Class9.3',
             'Class10.1',
             'Class10.2',
             'Class10.3',
             'Class11.1',
             'Class11.2',
             'Class11.3',
             'Class11.4',
             'Class11.5',
             'Class11.6' ]

def get_classes():
    return np.array( map( lambda x: int(x[len('Class'):-len('.1')] ),
                          get_fieldnames()[1:] ), dtype='int' )

def bbox( img, crop=False ):
    """
    Bounding box.
    """
    pts = np.transpose(np.nonzero(img>0))
    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0)
    if not crop:
        return ( x_min, y_min, x_max, y_max )
    else:
        return img[y_min:y_max+1,
                   x_min:x_max+1]

def saturate( img, q0=0.01, q1=0.99 ):
    """
    Saturate pixel intensities.
    """
    img = img.astype('float')
    if q0 is None:
        q0 = 0
    if q1 is None:
        q1 = 1
    q = mquantiles(img[np.nonzero(img)].flatten(),[q0,q1])
    img[img<q[0]] = q[0]
    img[img>q[1]] = q[1]
    return img
    
def rescale( img, min=0, max=255 ):
    """ Stretch contrast."""
    img = img.astype('float')
    img -= img.min()
    img /= img.max()
    img = img * (max - min) + min
    return img

def compose(m1,m2):
    n1 = np.eye(3,dtype='float32')
    n2 = np.eye(3,dtype='float32')
    n1[:2] = m1
    n2[:2] = m2
    n3 = np.dot(n1,n2)
    return n3[:2]

def rotate( img, (x,y), angle, interpolation=cv2.INTER_LINEAR ):
    """
    http://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    """
    cx = float(img.shape[1])/2
    cy = float(img.shape[0])/2    
    translation_mat = np.array([[1,0,cx-x],[0,1,cy-y]],dtype='float32')
    rotation_mat = cv2.getRotationMatrix2D((cx,cy),angle,1.0)
    m = compose( rotation_mat, translation_mat )
    res = cv2.warpAffine(img, m, img.shape[:2],flags=interpolation)
    return res

def recenter( img, (x,y), interpolation=cv2.INTER_LINEAR ):
    """
    http://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    """
    cx = float(img.shape[1])/2
    cy = float(img.shape[0])/2    
    translation_mat = np.array([[1,0,cx-x],[0,1,cy-y]],dtype='float32')
    res = cv2.warpAffine(img, translation_mat, img.shape[:2],flags=interpolation)
    return res

def unique_indices(M):
    """
    Removing duplicate rows
    http://mail.scipy.org/pipermail/scipy-user/2011-December/031193.html
    """
    unique_index = np.unique( M.dot(np.random.rand(M.shape[1])),
                                  return_index=True)[1]
    return unique_index

def fit_ellipse( points, factor=1.96, f="" ):
    """
    1.96 in order to contain 95% of the galaxy
    http://en.wikipedia.org/wiki/1.96
    """
    points = points.astype('float')

    center = points.mean(axis=0)
    points -= center
    # The singular values are already sorted in descending order.
    U, S, V = np.linalg.svd(points, full_matrices=False)

    if len(points)<10:
        print "ELLIPSE debug ", len(points), f
        
    S /= np.sqrt(len(points)-1)
    S *= factor
    angle = math.atan2(V[0,1],V[0,0])/math.pi*180
    
    return (center,2*S,angle)

def get_entropy(img):
    """
    http://stackoverflow.com/questions/16647116/faster-way-to-analyze-each-sub-window-in-an-image
    """
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    hist = hist.ravel()/hist.sum()
    logs = np.log2(hist+0.00001)
    return -1 * (hist*logs).sum()

def gini(x,f=""):
    """
    http://www.ellipsix.net/blog/2012/11/the-gini-coefficient-for-distribution-inequality.html
    """
    # requires all values in x to be zero or positive numbers,
    # otherwise results are undefined
    x = x.flatten()
    n = len(x)
    s = x.sum()
    r = np.argsort(np.argsort(-x)) # calculates zero-based ranks
    if s == 0 or n == 0:
        print "GINI debug",f
        return 1.0
    else:
        return 1.0 - (2.0 * (r*x).sum() + s)/(n*s)

def gaussian(shape):
    """Normalized 2D gaussian kernel"""
    x, y = np.mgrid[-shape[0]//2+1:shape[0]//2+1, -shape[1]//2+1:shape[1]//2+1]
    g = np.exp(-(x**2/(2.0*float(shape[0]//2//2)**2)+y**2/(2.0*float(shape[1]//2//2)**2)))
    return g / g.max()

def get_light_radius(img,r=[0.1,0.8]):
    img = img.astype('float')
    idx = np.nonzero(img)
    s = img[idx].sum()
    mask = np.ones(img.shape)
    mask[img.shape[0]/2,img.shape[1]/2] = 0
    edt = nd.distance_transform_edt( mask)
    edt[edt>=img.shape[1]/2] = 0
    edt[img==0] = 0
    q = mquantiles(edt[np.nonzero(edt)].flatten(),r)
    res = []
    for q0 in q:
        res.append( img[edt<q0].sum()/s )
    return res

def get_color_histogram(img_color):
    hist = np.array( [ np.bincount( img_color[:,:,0].flatten(),
                          minlength=256 ),
                       np.bincount( img_color[:,:,1].flatten(),
                          minlength=256 ),
                       np.bincount( img_color[:,:,2].flatten(),
                          minlength=256 ) ], dtype='float').flatten()
    
    # Normalize histogram
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm

    return hist 

def random_colors(labels):
    idx = np.nonzero(labels)
    nb_labels = labels.max()
    colors = np.random.random_integers(0,255,size=(nb_labels+1,3))
    seg = np.zeros( (labels.shape[0],labels.shape[1],3), dtype='uint8' )
    seg[idx] = colors[labels[idx].astype('int')]
    return seg
    
def largest_connected_component( img, labels, nb_labels ):
    """
    Select the largest connected component which is closest to the center
    using a weighting size/distance**2.
    """
    sizes =  np.bincount( labels.flatten(),
                          minlength=nb_labels+1 )
    centers = nd.center_of_mass( img, labels, range(1,nb_labels+1) )
    distances = map( lambda (y,x):
                         (img.shape[0]/2-y)**2+(img.shape[1]/2-x)**2,
                     centers )
    distances = [1.0] + distances
    distances = np.array(distances)
    sizes[0] = 0
    sizes[sizes<20] = 0
    sizes = sizes/(distances+0.000001)
    best_label = np.argmax( sizes )
    thresholded = (labels==best_label)*255
    
    return thresholded

def get_features( f,
                  dsift_learn=False,
                  debug=False,
                  tiny_img=False,
                  tiny_grad=False,
                  image_statistics=False,
                  color_histogram=False,
                  orientation_histogram=False,
                  transform=0 ):
    img_color = cv2.imread( f ).astype('float')
    if transform == 1:
        img_color = img_color[::-1,:].copy()
    if transform == 2:
        img_color = img_color[:,::-1].copy()

    original_img = img_color.copy()
    # denoise
    img_color = cv2.GaussianBlur(img_color,(0,0),2.0)
    
    img = cv2.cvtColor( img_color.astype('uint8'),
                        cv2.COLOR_BGR2GRAY ).astype('float')
    
    # TEDYouth 2013 - Filmed November 2013 - 6:43
    # Henry Lin: What we can learn from galaxies far, far away
    # "I subtract away all of the starlight"
    #t = img[np.nonzero(img)].mean()
    #t = img.mean()
    #t = np.max(img_color[np.nonzero(img)].mean(axis=0))
    t = np.max(np.median(img_color[np.nonzero(img)],axis=0))
    
    img_color[img_color<t] = t
    img_color = rescale(img_color).astype('uint8')
    
    if debug:
        cv2.imwrite("start.png",img_color)
        
    img = cv2.cvtColor( img_color.astype('uint8'),
                        cv2.COLOR_BGR2GRAY )

    saturated = saturate(img,q1=0.75)
    labels,nb_maxima = nd.label(saturated==saturated.max(), output='int64')

    if debug:
        cv2.imwrite("adaptive_labels.png",random_colors(labels))
    
    thresholded = largest_connected_component( img, labels, nb_maxima )
    center = nd.center_of_mass( img, thresholded )

    # features from original image
    original_size = img_color.shape[0]*img_color.shape[1]
    original_shape = img_color.shape[:2]

    idx = np.nonzero(original_img.mean(axis=2))
    nonzero = original_img[idx]
    mean = nonzero.mean(axis=0)
    std = nonzero.std(axis=0)
    mean_center = original_img[thresholded>0].mean(axis=0)
    features = np.array( [ mean_center[0],mean_center[1],mean_center[2],
                           mean[0],mean[1],mean[2],
                           std[0],std[1],std[2],
                           kurtosis(nonzero[:,0]),
                           kurtosis(nonzero[:,1]),
                           kurtosis(nonzero[:,2]),
                           skew(nonzero[:,0]),
                           skew(nonzero[:,1]),
                           skew(nonzero[:,2]),
                           gini(nonzero[:,0],f),
                           gini(nonzero[:,1],f),
                           gini(nonzero[:,1],f)
                           ], dtype='float' )
    # features = np.empty( 0, dtype='float' )
    
    img_color = recenter( img_color, (center[1],center[0]), interpolation=cv2.INTER_LINEAR )

    img = cv2.cvtColor( img_color.astype('uint8'),
                        cv2.COLOR_BGR2GRAY )

    # offset from center
    center_offset = np.linalg.norm(np.array([center[0],center[1]],dtype='float')
                                   - np.array(original_shape,dtype='float')/2)
            
    
    # adaptive thresholding
    thresholded = cv2.adaptiveThreshold( img,
                                         maxValue=255,
                                         adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         thresholdType=cv2.THRESH_BINARY,
                                         blockSize=301,
                                         C=0 )

    # select largest connected component
    # which is closest to the center
    # use a weighting size/distance**2
    labels, nb_labels = nd.label( thresholded, output='int64' )

    if debug:
        cv2.imwrite("debug.png",random_colors(labels))
        
    thresholded = largest_connected_component( img, labels, nb_labels )
        
    if debug:
        cv2.imwrite( "debug_tresholded.png", thresholded )
    
    # ellipse fitting
    # cv2.fitEllipse returns a tuple.
    # Tuples are immutable, they can't have their values changed.
    # we have the choice between the least-square minimization implemented in
    # OpenCV or a plain PCA.
    XY = np.transpose( np.nonzero(thresholded) )[:,::-1]
    #((cx,cy),(w,h),angle) = cv2.fitEllipse(XY)
    # eccentricity = math.sqrt(1-(np.min((w,h))/np.max((w,h)))**2)   
    ((cx,cy),(w,h),angle) = fit_ellipse(XY,f=f)
    #eccentricity = math.sqrt(1-(h/w)**2)
    eccentricity = h/w
    
    if w == 0 or h == 0:
        print "bad ellipse:", ((cx,cy),(w,h),angle), f
        exit(1)

    if debug:
        print ((cx,cy),(w,h),angle)

    # rotate image
    img_color = rotate( img_color, (cx,cy), angle, interpolation=cv2.INTER_LINEAR )

    thresholded = rotate( thresholded, (cx,cy), angle, interpolation=cv2.INTER_NEAREST )

    cx = float(img.shape[1])/2
    cy = float(img.shape[0])/2        
    
    if debug:
        cv2.imwrite( "rotated.png", img_color )

    # crop
    img_color = img_color[max(0,int(cy-h/2)):min(img.shape[0],int(cy+h/2+1)),
                          max(0,int(cx-w/2)):min(img.shape[1],int(cx+w/2+1))]
    thresholded = thresholded[max(0,int(cy-h/2)):min(img.shape[0],int(cy+h/2+1)),
                              max(0,int(cx-w/2)):min(img.shape[1],int(cx+w/2+1))]

    if debug:
        cv2.imwrite("cropped_thresholded.png",thresholded)

    color_hist = get_color_histogram(img_color)
    if color_histogram:
        return color_hist

    if orientation_histogram:
        return get_orientation_histogram(img_color)    

    img = cv2.cvtColor( img_color, cv2.COLOR_BGR2GRAY ).astype('float')
    img = rescale(img)

    saturated = saturate(img,q1=0.95)
    labels,nb_maxima = nd.label(saturated==saturated.max(), output='int64')

    if debug:
        cv2.imwrite("labels.png",random_colors(labels))
    
    if img_color.shape[0] == 0 or img_color.shape[1] == 0:
        print "bad size", img_color.shape, f
        exit(1)

    img_thumbnail = cv2.resize( img.astype('uint8'), (64,64),
                                interpolation=cv2.INTER_AREA )

    if tiny_img:
            return img_thumbnail.flatten()

    if debug:
        cv2.imwrite( "tiny_img.png", img_thumbnail )     

    grad_color = nd.gaussian_gradient_magnitude(img_color,1.0)
    grad_img = rescale(img_color[:,:,0]+img_color[:,:,2])

    if debug:
        cv2.imwrite( "channel.png", grad_img )        

    grad_thumbnail = cv2.resize( grad_img, (64,64),
                                 interpolation=cv2.INTER_AREA )

    if debug:
        cv2.imwrite( "tiny_grad.png", grad_thumbnail )
            
    if tiny_grad == True:
        # return np.append( [eccentricity*100],
        #                   grad_thumbnail.flatten() )
        return grad_thumbnail.flatten()

    if debug:
        cv2.imwrite( "cropped.png", img_color )

    # chirality
    # http://en.wikipedia.org/wiki/Chirality
    # An object is chiral if it is not identical to its mirror image.
    #mirror_spiral_img = labels[:,::-1]
    mirror_grad_img = grad_img[:,::-1]
    chirality = np.sum(np.sqrt( (grad_img - mirror_grad_img)**2 ))/ (grad_img.sum()) 

    # compare size of the thresholded area to the size of the fitted ellipse
    # and to the size of the whole image
    size_to_ellipse = float(thresholded.sum()) / (math.pi * w * h / 4)
    box_to_image = float(img.shape[0]*img.shape[1]) / original_size

    if size_to_ellipse < 0.1:
        print "SIZE_TO_ELLIPSE debug", f
        
    if box_to_image > 0.5:
        print "BOX_TO_IMAGE debug", f
        
    # color features
    # central pixel and mean channel values
    idx = np.nonzero(thresholded)
    mean = img_color[idx].mean(axis=0)
    grey_mean = img[idx].mean()
    img_center = img[img.shape[0]/2-img.shape[0]/4:img.shape[0]/2+img.shape[0]/4,
                     img.shape[1]/2-img.shape[1]/4:img.shape[1]/2+img.shape[1]/4]
    img_center_color = img_color[img.shape[0]/2-img.shape[0]/4:img.shape[0]/2+img.shape[0]/4,
                                 img.shape[1]/2-img.shape[1]/4:img.shape[1]/2+img.shape[1]/4]
    center_mean = img_center[np.nonzero(img_center)].mean()
    center_mean_color = img_center_color[np.nonzero(img_center)].mean(axis=0)
    color_features = [
        img_color[img_color.shape[0]/2,
                                 img_color.shape[1]/2,0],
                       img_color[img_color.shape[0]/2,
                                 img_color.shape[1]/2,1],
                       img_color[img_color.shape[0]/2,
                                 img_color.shape[1]/2,2],
                      mean[0],mean[1],mean[2],
                       center_mean_color[0],center_mean_color[1],center_mean_color[2],
                       float(img[img.shape[0]/2,
                                 img.shape[1]/2])/grey_mean,
                       float(center_mean)/grey_mean]

    entropy = get_entropy(img.astype('uint8'))

    light_radius = get_light_radius(img)
    features = np.append( features, [ eccentricity,
                                      w,h,
                                      thresholded.sum(),
                                     entropy,
                                     chirality,
                                     size_to_ellipse,
                                     box_to_image,
                                     center_offset,
                                     light_radius[0],
                                      light_radius[1],
                                      nb_maxima,
                                      kurtosis(img_color[idx][:,0]),
                                      kurtosis(img_color[idx][:,1]),
                                      kurtosis(img_color[idx][:,2]),
                                      skew(img_color[idx][:,0]),
                                      skew(img_color[idx][:,1]),
                                      skew(img_color[idx][:,2]),
                                      gini(img_color[idx][:,0],f),
                                      gini(img_color[idx][:,1],f),
                                      gini(img_color[idx][:,2],f),
                                      kurtosis(img[idx]),
                                      skew(img[idx]),
                                      gini(img[idx],f)
                                      ] )    
    features = np.append( features, color_features )
                           
    # Hu moments from segmentation
    m = cv2.moments( thresholded.astype('uint8' ), binaryImage=True )
    hu1 = cv2.HuMoments( m )

    # Hu moments from taking pixel intensities into account
    m = cv2.moments( img, binaryImage=False )
    hu2 = cv2.HuMoments( m )
    
    m = cv2.moments( grad_img, binaryImage=False )
    hu3 = cv2.HuMoments( m )    

    hu = np.append( hu1.flatten(), hu2.flatten() )
    hu = np.append( hu.flatten(), hu3.flatten() )
    features = np.append( features, hu.flatten() )

    features = np.append( features, hu )

    if image_statistics:
        return features

    # features = np.empty( 0, dtype='float' )

    average_prediction = np.zeros( 37, dtype='float' )
    
    # PCA features
    if not debug:
        image_statistics = features
        for Class in xrange(1,12):
            scaler = joblib.load(get_data_folder()+"/scaler_statistics_Class"+ str(Class)+"_")
            clf = joblib.load(get_data_folder()+"/svm_statistics_Class"+ str(Class)+"_")
            features = np.append( features, clf.predict_proba(scaler.transform(image_statistics)))

        average_prediction += features[-37:]
        
        grad_thumbnail = grad_thumbnail.flatten()
        for Class in xrange(1,12):
            pca = joblib.load(get_data_folder()+"/pca_Class"+ str(Class)+"_")
            thumbnail_pca = pca.transform(grad_thumbnail)

            clf = joblib.load(get_data_folder()+"/pca_SVM_Class" + str(Class)+"_")
            features = np.append( features,
                                  clf.predict_proba(thumbnail_pca).flatten() )

        average_prediction += features[-37:]
            
        img_thumbnail = img_thumbnail.flatten()
        for Class in xrange(1,12):
            pca = joblib.load(get_data_folder()+"/pca_img_Class"+ str(Class)+"_")
            thumbnail_pca = pca.transform(img_thumbnail)

            clf = joblib.load(get_data_folder()+"/pca_img_SVM_Class" + str(Class)+"_")
            features = np.append( features,
                                  clf.predict_proba(thumbnail_pca).flatten() )

        average_prediction += features[-37:]
        
        for Class in xrange(1,12):
            pca = joblib.load(get_data_folder()+"/pca_color_Class"+ str(Class)+"_")
            hist_pca = pca.transform(color_hist)

            clf = joblib.load(get_data_folder()+"/pca_color_SVM_Class" + str(Class)+"_")
            features = np.append( features,
                                  clf.predict_proba(hist_pca).flatten() )            

        average_prediction += features[-37:]
        average_prediction /= 4
        features = np.append( features, average_prediction )
        
    return features

def read_responses( f ):
    reader = csv.DictReader( open( f, 'rb' ) )
    responses = []
    ids = []
    for line in reader:
        ids.append( line['GalaxyID'] )
        r = []
        for k in get_fieldnames():
            if k == 'GalaxyID':
                continue
            r.append( float(line[k]) )
        responses.append( r )
    return np.array(responses,dtype='float32'),np.array(ids)

def read_responses_as_dict( f ):
    reader = csv.DictReader( open( f, 'rb' ) )
    responses = []
    for line in reader:
        responses.append( line )
    return responses

if __name__ == '__main__':
    f = sys.argv[1]

    print get_features(f,debug=False,tiny_img=False)

