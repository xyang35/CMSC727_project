from imdb import load_data, prepare_data
import numpy as np
import pickle as pkl
train, valid, test = load_data(n_words=10, valid_portion=0.05)
x = [train[0][t]for t in range(0,len(train[0]))]
y = [train[1][t]for t in range(0,len(train[1]))]
x, mask, y = prepare_data(x, y)
y = np.array(y)
feat_train = np.zeros((x.shape[0],x.shape[1],10))
for i in range(0, x.shape[0]):
    print "num: "+str(i)
    for j in range(0, x.shape[1]):
        feat_train[i][j][x[i][j]]=1
np.save("data/feats_train.npy",feat_train)
np.save("data/labels_train.npy",y)
np.save("data/mask_train.npy",mask)






