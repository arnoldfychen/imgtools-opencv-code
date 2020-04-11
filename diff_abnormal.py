# encoding: utf-8
'''
 fychen  Aug/2019
'''
import os
from skimage.measure import compare_ssim
import imutils
import cv2
import numpy as np
import glob
import math

MAX_MATCHES = 500
GOOD_MATCH_PERCENT = 0.15

'''
def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz,1);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

def dehaze(src):
    I = src.astype('float64')/255;
    dark = DarkChannel(I,15);
    #cv2.imshow("dark",dark)
    A = AtmLight(I,dark);
    te = TransmissionEstimate(I,A,15);
    t = TransmissionRefine(src,te);
    R = Recover(I,t,A,0.1)
    R = R*255;
    file="test_dh.jpg"
    cv2.imwrite(file, R);
    return file
'''
    
def getprob(gray_img):
    prob={}
    height,width = gray_img.shape
    for i in range(0,height):
        for j in range(0,width):
            pixel = gray_img[i,j]
            #if(pixel)
            if pixel in prob:
               prob[pixel] += 1
            else:
               prob[pixel] = 1
    pixnum =  width *height
    for i in range(0,256):        
        prob[i] = prob[i] /pixnum
    #print(prob)
    return prob

def get_perfect_threshold(prob):
    threshold = 0
    maxf = 0.0
    for cod in range(1,256):
        W0 = 0.0
        W1 = 0.0
        U0 = 0.0
        U1 = 0.0

        for i in range(0,cod+1):
            U0 += i * prob[i]       
            W0 += prob[i]

        for x in range(cod,256):
            U1 += i * prob[x]
            W1 += prob[x]
      
        if W0 == 0 or W1 == 0:
            continue
        U0 /= W0
        U1 /= W1
        D0 = 0.0
        D1= 0.0
 
        for i in range(0,cod+1):
            D0 += pow((i-U0)*prob[i],2.0)

        for x in range(cod,256):
            D1 += pow((i-U1)*prob[i], 2.0)
        D0 /= W0
        D1 /= W1
        Dw = pow(D0, 2.0) * W0 + pow(D1, 2.0) * W1
        Db = W0 * W1 * pow((U1-U0), 2.0)
        
        f = 1.0*Db/(Db+Dw)
        #print("W0={},W1={},U0={},U1={},D0={},D1={},Db={},f={},maxf={}".format(W0,W1,U0,U1,D0,D1,Db,f,maxf))
        print("f: %f, maxf: %f, threshold %d,cod %d" %(f,maxf,threshold,cod))
        if maxf*1.0 - f < 0.001:
            maxf = f
            threshold = cod
    return threshold
 
def alignImages(im1, im2):

  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
  
  '''
  # Test result shows SURF is not better than ORB, so disable it!  fychen
  #SURF
  surf = cv2.xfeatures2d.SURF_create(MAX_MATCHES)
  keypoints1, descriptors1 = surf.detectAndCompute(im1Gray, None) 
  keypoints2, descriptors2 = surf.detectAndCompute(im2Gray, None)
  
  bf = cv2.BFMatcher(cv2.NORM_L2)
  matches = bf.match(descriptors1, descriptors2)
  src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
  dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
  h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
  '''
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_MATCHES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
  
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
  
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
  
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
  
  
  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
  
  return im1Reg, h

def detect(ref_img,match_img,output_path):
    filename_full = os.path.basename(match_img)
    filename = filename_full.split(".")[0]
    imageA = cv2.imread(ref_img)
    imageB = cv2.imread(match_img)
    #cv2.imencode('.jpg',match_img)[1].tofile(filename)
    if imageA is None or imageB is None:
        imageA = cv2.imdecode(np.fromfile(ref_img,dtype=np.uint8),-1)
        imageB = cv2.imdecode(np.fromfile(match_img,dtype=np.uint8),-1)

    hA= imageA.shape[0]
    wA= imageA.shape[1]
    hB= imageB.shape[0]
    wB= imageB.shape[1]
    if hA!=hB or wA!=wB:
        imageB = cv2.resize(imageB,(wA,hA), interpolation = cv2.INTER_CUBIC)

    #print("height=%d, width=%d" %(hA,wA))
    #tomatch=1st, reference=2nd
    #cv2.imshow("imageB",imageB)
    imMatched, h = alignImages(imageB, imageA)
    #cv2.imshow("imMatched",imMatched)
    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imMatched, cv2.COLOR_BGR2GRAY)
    
    #grayA=cv2.adaptiveThreshold(grayA,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,155,1)
    #grayB=cv2.adaptiveThreshold(grayA,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,155,1)
    
    
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    #(score, diff) = compare_ssim(imageA, imMatched, full=True, multichannel=True)
    diff = (diff * 255).astype("uint8")
    #print("SSIM: {}".format(score))
    
    
    '''
    imdiff = cv2.subtract(grayB,grayA);
    diff = abs(imdiff);
    diff = diff.astype("uint8")
    '''
    #print(diff[100,100])
    #diff.show()
    #cv2.imshow("diff",diff)
    
    '''
    prob = getprob(diff)
    perf_threshold = get_perfect_threshold(prob)
    #print("perf_threshold: %d" %perf_threshold)
    thresh = cv2.threshold(diff, perf_threshold, 255,cv2.THRESH_BINARY)[1]
    '''
    thresh = cv2.threshold(diff, 100, 255,cv2.THRESH_BINARY_INV)[1]
    thresh = cv2.medianBlur(thresh,5)
    struct = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3),(-1,-1))
    ##thresh = cv2.erode(thresh,struct)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, struct,0,(-1,-1),2,cv2.BORDER_REPLICATE)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #cv2.CHAIN_APPROX_SIMPLE
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    
    areas = {}
    max_area=0
    x_min=0
    y_min=0
    x_max=0
    y_max=0
    
    for c in cnts:
        # cc = cv2.approxPolyDP(c,0.00001 * cv2.arcLength(c,True),True)
        x_min, y_min, w, h = cv2.boundingRect(c)
        #print("(x=%d,y=%d),w=%d,h=%d,area=%d" %(x_min,y_min,w,h,w*h))
        x_max = x_min+w
        y_max = y_min+h
        area = w*h
        max_area = max(area,max_area)
        areas[area]=[x_min,y_min,x_max,y_max]
        #if x_min==0 or x_max == wA or y_min==0 or y_max ==hA:
        #    continue
    
    f = open(output_path+os.path.sep+filename+".txt","w")
    f.writelines("ID,PATH,TYPE,XMIN,YMIN,XMAX,YMAX\n")
    if max_area >0:
        rect = areas[max_area]
        x_min=rect[0]
        y_min=rect[1]
        x_max=rect[2]
        y_max=rect[3]
        
        cv2.rectangle(imageB, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        f.writelines("1,"+filename_full+",1,"+str(x_min)+","+str(y_min)+","+str(x_max)+","+str(y_max))
    else:
        f.writelines("1,"+filename_full+",0,0,0,0,0")
    f.close()
        
    # show the output images
    cv2.imshow("Original", imageA)
    cv2.imshow("Modified", imageB)
    #cv2.imshow("Diff", diff)
    #cv2.imshow("Thresh", thresh)
    #cv2.imshow("contours",gray_diff)
    cv2.waitKey(0)
    return x_min,y_min,x_max,y_max

def defect_diff(input_path,output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    normal_files = glob.glob(input_path+os.path.sep+"*normal.jpg")
    normal_files2 = glob.glob(input_path+os.path.sep+"*normal.JPG")
    normal_files.extend(normal_files2)
    #print("normal_files= " %normal_files)
    normal_files.sort()
    for normal_file in normal_files:
        filename = os.path.basename(normal_file)
        suffix = filename.split(".")[1].lower()
        group_id = filename[0:4]
        #print("group_id %s" %group_id)
        todetect_files = glob.glob(input_path+os.path.sep+group_id+"_[0-9]."+suffix)
        todetect_files2 = glob.glob(input_path+os.path.sep+group_id+"_[0-9]."+suffix.upper())
        todetect_files.extend(todetect_files2)
        for detect_file in todetect_files:
            print("nomal_file=%s,detect_file=%s" %(normal_file,detect_file))
            detect(normal_file,detect_file,output_path)
    
if __name__ == '__main__' :
    output_path="bj_output"
    input_path="input"
    defect_diff(input_path,output_path)