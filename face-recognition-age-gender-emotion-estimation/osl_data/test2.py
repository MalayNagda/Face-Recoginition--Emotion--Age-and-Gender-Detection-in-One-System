database = {}
path='E:/osl_data/images/'
     
for f in listdir(path):     
    if f.startswith('.'):
        continue
    print(f)
    # Iterate over index 
    name=''
    for element in range(0, len(f)): 
        if f[element]=='.':
            break
        name+=f[element]
    print(name)

        #if f is null
    database[name] = img_to_encoding(path+f) 


for f in listdir(path):     
        if f.startswith('.'):
            continue
        print(f)