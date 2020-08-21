import os
from os import listdir

database = {}
path='E:/osl_data/images/'
for f in listdir(path):     
    if f.startswith('.'):
        continue
    # Iterate over index 
    #print(f)
    name=''
    for element in range(0, len(f)): 
        if f[element]=='.':
            break
        name+=f[element]
    print(name)
        
     database['name'] = img_to_encoding('path_add')
           
              