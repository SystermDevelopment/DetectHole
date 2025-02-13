import os

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

def network(x, y, test=False):
    # Input:x -> 1,96,49
    # MulScalar_2
    h = F.mul_scalar(x, val=0.003921568627451001)
    # ImageAugmentation
    if not test:
        h = F.image_augmentation(h, shape=(1,96,49), pad=(2,2), min_scale=0.94, max_scale=1.04, angle=0, aspect_ratio=1, distortion=0.1, brightness=0, contrast=1, noise=0)
    # BinaryConnectAffine -> 36
    h = PF.binary_connect_affine(h, (36,), name='BinaryConnectAffine')
    # PReLU
    h = PF.prelu(h, base_axis=1, shared=False, name='PReLU')
    # BinaryConnectAffine_2 -> 2048
    h = PF.binary_connect_affine(h, (2048,), name='BinaryConnectAffine_2')
    # ELU
    h = F.elu(h, alpha=1)
    # BinaryConnectAffine_3 -> 22701
    h = PF.binary_connect_affine(h, (22701,), name='BinaryConnectAffine_3')
    # BatchNormalization_3
    h = PF.batch_normalization(h, decay_rate=0.5, eps=0.01, batch_stat=not test, name='BatchNormalization_3')
    # ReLU_3
    h = F.relu(h, inplace=True)
    # BinaryConnectAffine_4 -> 49
    h = PF.binary_connect_affine(h, (49,), name='BinaryConnectAffine_4')
    # BatchNormalization_4
    h = PF.batch_normalization(h, decay_rate=0.5, eps=0.01, batch_stat=not test, name='BatchNormalization_4')
    # Softmax
    h = F.softmax(h)
    # CategoricalCrossEntropy -> 1
    #h = F.categorical_cross_entropy(h, y)
    return h

def network2(x, y, test=False):
    # Input:x -> 1,96,49
    # ImageAugmentation
    if not test:
        h = F.image_augmentation(x, shape=(1,96,49), pad=(2,2), min_scale=0.94, max_scale=1.04, angle=0, aspect_ratio=1, distortion=0.1, brightness=0, contrast=1, noise=0)
    else:
        h = x
    # BinaryConnectAffine -> 36
    h = PF.binary_connect_affine(h, (36,), name='BinaryConnectAffine')
    # PReLU
    h = PF.prelu(h, base_axis=1, shared=False, name='PReLU')
    # BinaryConnectAffine_2 -> 2048
    h = PF.binary_connect_affine(h, (2048,), name='BinaryConnectAffine_2')
    # ELU
    h = F.elu(h, alpha=1)
    # BinaryConnectAffine_3 -> 22701
    h = PF.binary_connect_affine(h, (22701,), name='BinaryConnectAffine_3')
    # BatchNormalization_3
    h = PF.batch_normalization(h, decay_rate=0.5, eps=0.01, batch_stat=not test, name='BatchNormalization_3')
    # ReLU_3
    h = F.relu(h, inplace=True)
    # BinaryConnectAffine_4 -> 49
    h = PF.binary_connect_affine(h, (49,), name='BinaryConnectAffine_4')
    # BatchNormalization_4
    h = PF.batch_normalization(h, decay_rate=0.5, eps=0.01, batch_stat=not test, name='BatchNormalization_4')
    # Softmax
    h = F.softmax(h)
    # CategoricalCrossEntropy -> 1
    #h = F.categorical_cross_entropy(h, y)
    return h

def network3(x, y, test=False):
    # Input:x -> 1,96,49
    # ImageAugmentation
    if not test:
        h = F.image_augmentation(x, shape=(1,96,49), pad=(2,2), min_scale=0.94, max_scale=1.04, angle=0, aspect_ratio=1, distortion=0.1, brightness=0, contrast=1, noise=0)
    else:
        h = x
    # BinaryConnectAffine -> 36
    h = PF.binary_connect_affine(h, (36,), name='BinaryConnectAffine')
    # PReLU
    h = PF.prelu(h, base_axis=1, shared=False, name='PReLU')
    # BinaryConnectAffine_2 -> 2048
    h = PF.binary_connect_affine(h, (1024,), name='BinaryConnectAffine_2')
    # ELU
    h = F.elu(h, alpha=1)
    # BinaryConnectAffine_3 -> 22701
    h = PF.binary_connect_affine(h, (500,), name='BinaryConnectAffine_3')
    # BatchNormalization_3
    h = PF.batch_normalization(h, decay_rate=0.5, eps=0.01, batch_stat=not test, name='BatchNormalization_3')
    # ReLU_3
    h = F.relu(h, inplace=True)
    # BinaryConnectAffine_4 -> 49
    h = PF.binary_connect_affine(h, (49,), name='BinaryConnectAffine_4')
    # BatchNormalization_4
    h = PF.batch_normalization(h, decay_rate=0.5, eps=0.01, batch_stat=not test, name='BatchNormalization_4')
    # Softmax
    h = F.softmax(h)
    # CategoricalCrossEntropy -> 1
    #h = F.categorical_cross_entropy(h, y)
    return h

class AngleNetwork:
    def create(type=0):
        cur = os.path.dirname(__file__)
        #cur = "F:/nnc_projects/ring_rotate_detect2.files/"

        with nn.parameter_scope('angle'):
            #nn.load_parameters(cur + '/holeexist/20230225_191153/results.nnp')

            params = [
                ['20230804_123221', network],       # 0
                ['20230804_142131', network2],      # 1
                ['20230805_185220', network3],      # 2
                ['', None],
            ]

            nn.load_parameters(cur + '/angle/' + params[type][0] +  '/results.nnp')
            #nn.load_parameters(cur + params[type][0] +  '/results.nnp')

            x = nn.Variable((1, 1, 96, 49))
            y = params[type][1](x, 1, True)

        return x, y


if __name__ == "__main__":
    import cv2 as cv
    import numpy as np

    x, y = AngleNetwork.create(1)

    img = cv.imread("F://nnc_datasets/ring_angle_detect_1/dataset/004/-40_103419_021225.png", cv.IMREAD_UNCHANGED)
    x.d = img.reshape(1,1,96,49)
    #x.d = img
    y.forward()
    print("get_angle %d" % (int(np.argmax(y.d) - 48)))
