import os 
import shutil
import random

origPath = r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\negative'
numTrain = 500
numTest = 100
numVal = 100

listMove = random.choices(os.listdir(origPath), k = (numTrain + numTest + numVal)* 7)
# print(listMove)
for name in listMove:
    subset = random.choice(['train', 'test', 'valid'])
    orig = os.path.join(origPath, name)
    if subset == 'train':
        if numTrain == 0: 
            continue
        destination = os.path.join(r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\mini-zalo-data\train\7', name)
        shutil.copy(orig, destination)
        numTrain -= 1
    elif subset == 'test':
        if numTest ==0:
            continue
        destination = os.path.join(r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\mini-zalo-data\test\7', name)
        numTest -=1
        shutil.copy(orig, destination)
    elif subset == 'valid':
        if numVal == 0:
            continue
        destination = os.path.join(r'C:\Users\QuangDuy\Desktop\bienbao_data\BTL_AI\mini-zalo-data\valid\7', name)
        numVal -= 1
        shutil.copy(orig, destination)
    print('done {}'.format(name))