import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import cv2
import numpy as np

#np.set_printoptions(suppress=True)

def network(x, y, test=False):
    # Input:x -> 3,74,768
    # MaxPooling -> 3,1,96
    h = F.max_pooling(x, (57, 8), (57, 8))
    # ReLU_2
    h = F.relu(h, True)
    # MaxPooling_2
    h = F.max_pooling(h, (424, 1), (424, 1))
    # ReLU
    h = F.relu(h, True)
    # AveragePooling_2
    h = F.average_pooling(h, (1, 1), (1, 1))
    # Swish
    h = F.swish(h)
    # Affine -> 1
    h = PF.affine(h, (1,), name='Affine')
    # BatchNormalization
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
    # Sigmoid
    h = F.sigmoid(h)
    # BinaryCrossEntropy
    #h = F.binary_cross_entropy(h, y)
    return h

def imread(path):
    img_bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.transpose(2, 0, 1)

    return img

if __name__ == "__main__":
    # load parameters
    nn.load_parameters('../ml_wirering.files/20200328_224935/results.nnp')

    # Prepare input variable
    x = nn.Variable((1, 3, 74, 768))
    # Build network for inference
    y = network(x, 1, test=True)

    img_ok = imread('../data3/dataset/ok/10185.png')
    img_ng = imread('../data3/dataset/ng/img10233.png')

    x.d = img_ok.reshape(x.shape)
    y.forward()
    print('ok ' + str(y.d))     # 1に近ければOK

    x.d = img_ng.reshape(x.shape)
    y.forward()

    print('ng ' + str(y.d))     # 0に近ければOK
