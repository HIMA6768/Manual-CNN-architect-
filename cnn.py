import numpy as np 
from sampleinput import img
from kernels import emboss

kernel= emboss

def addpadding(img,p):
    img=np.array(img)
    for i in range(p) :
        img=np.insert(img,i,0,axis=0)
        img=np.insert(img,img.shape[0],0,axis=0)
        img=np.insert(img,i,0,axis=1)
        img=np.insert(img,img.shape[1],0,axis=1)
    return img

def convolution(arr,kernel,s):
    f=kernel.shape[0]
    p=int((f-1)/2)
    arr=addpadding(arr,p)
    rs=0
    sh=kernel.shape[0]
    output=[]
    while(rs<=arr.shape[0]-kernel.shape[0]):
        cs=0
        while(cs<=arr.shape[1]-kernel.shape[1]):
            mat=arr[rs:rs+sh,cs:cs+sh]
            # print(mat)
            output.append(np.sum(mat*kernel).item())
            cs+=s
        rs+=s
    sizer=int((arr.shape[0]-f)/s+1)
    sizec=int((arr.shape[1]-f)/s+1)
    output=np.array(output).reshape(sizer,sizec)
    output = np.round(output, 2)
    # print(f"result = \n {output}")
    return output

def relu(arr):
 return np.maximum(0,arr)
   
 
def maxpooling(arr,s):
    rs=0
    sh=s
    output=[]
    while(rs<=arr.shape[0]-s):
        cs=0
        while(cs<=arr.shape[1]-s):
            mat=arr[rs:rs+sh,cs:cs+sh]
            output.append(np.max(mat).item())
        
            cs+=s
        rs+=s
    os=int(np.sqrt(len(output)))
    return np.array(output).reshape(os,os)   


def minpooling(arr,s):
    rs=0
    sh=s
    output=[]
    
    while(rs<=arr.shape[0]-s):
        cs=0
        while(cs<=arr.shape[1]-s):
            mat=arr[rs:rs+sh,cs:cs+sh]
            output.append(np.min(mat).item())
            
        
            cs+=s
        rs+=s
    os=int(np.sqrt(len(output)))
    return np.array(output).reshape(os,os)   

def avgpooling(arr,s):
    rs=0
    sh=s
    output=[]
    while(rs<=arr.shape[0]-s):
        cs=0
        while(cs<=arr.shape[1]-s):
            mat=arr[rs:rs+sh,cs:cs+sh]
            output.append(np.average(mat).item())
            cs+=s
        rs+=s
    os=int(np.sqrt(len(output)))
    return np.array(output).reshape(os,os)   


def rgbconvolution(img,kernel,stride):
    
    s=stride
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]

    r=np.array(convolution(r,kernel,s))
    
    g=np.array(convolution(g,kernel,s))

    b=np.array(convolution(b,kernel,s))
    
    return r+g+b


def flatten(p):
    return np.ravel(p)


# c=rgbconvolution(img,kernel,1)
# print(f" final convoluted result =\n {c}")
# print(relu(c))
