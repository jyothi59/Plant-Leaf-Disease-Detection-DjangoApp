from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
img_w,img_h= 224,224
label=['Bacterial spot', 'Early blight', 'healthy', 'Late blight', 'Leaf Mold', 'Septoria leaf spot', 'Spider mites Two-spotted spider mite', 'Target Spot', 'Tomato mosaic virus', 'Yellow Leaf Curl Virus']
model=load_model('./models/vgg16(7).h5')


# Create your views here.
def index(request):
    context={'a':1}
    return render(request,'index.html',context)
def tomato(request):
    context={'a':1}
    return render(request,'tomato.html',context)
def about(request):
    context={'a':1}
    return render(request,'about.html',context)
def contact(request):
    context={'a':1}
    return render(request,'contact.html',context)
def predictImage(request):
    print(request)
    print(request.POST.dict())
    print(3)
    imgObj=request.FILES['filePath']
    print(4)
    print(imgObj)
    fs=FileSystemStorage()
    imgPath=fs.save(imgObj.name,imgObj)
    imgPath=fs.url(imgPath)
    testimage='.'+imgPath
    img =image.load_img(testimage,target_size=(img_w,img_h))
    xt = img_to_array(img)
    xt=xt/255
    xt=xt.reshape(1,img_w,img_h,3)
    predi=model.predict(xt)
    predict=label[np.argmax(predi)]
    

    context={"imgPath":imgPath,"class":predict}
  
    return render(request,'result.html',context)