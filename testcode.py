from sampleinput import img
from kernels import emboss
from cnn import rgbconvolution,convolution,relu,maxpooling,flatten

image=img
kernel=emboss

# print(image) 

convolution1= rgbconvolution(image,kernel,1)
print(f"after convolution1 : \n {convolution1}")
relu1=relu(convolution1)
print(f"after relu1 : \n {relu1}")
maxpooling1=maxpooling(relu1,2)
print(f"after maxpooling1 : \n {maxpooling1}")

convolution2=convolution(maxpooling1,kernel,1)
print(f"after convolution2 : \n {convolution2}")
relu2=relu(convolution2)
print(f"after relu2 : \n {relu2}")

flatten1=flatten(relu2)
print(f" use this flattenarray in your newral network : \n{flatten1}")




