import cv2
# importing os module   
import os 
  

#image directory
directory = r'E:/osl_data/'
# Change the current directory  
# to specified directory  
os.chdir(directory) 

print("Before saving image:")   
print(os.listdir(directory))   
  
  
camera = cv2.VideoCapture(0)
for i in range(1):
    return_value, image = camera.read()
    cv2.imwrite('opencv'+str(i)+'.png', image)
del(camera)