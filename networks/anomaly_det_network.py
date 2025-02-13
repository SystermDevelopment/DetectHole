import os

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

def network(x, y, test=True):
    # Input:x -> 3,625,145
    # AveragePooling -> 3,312,72
    h = F.average_pooling(x, (2,2), (2,2))
    # Convolution -> 17,312,72
    h = PF.convolution(h, 17, (3,3), (1,1), name='Convolution')
    # BatchNormalization
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
    # AveragePooling_2 -> 17,156,18
    h = F.average_pooling(h, (2,4), (2,4))
    # Affine -> 2
    h = PF.affine(h, (2,), name='Affine')
    # Softmax
    h = F.softmax(h)
    # CategoricalCrossEntropy -> 1
    #h = F.categorical_cross_entropy(h, y)
    return h

def network2(x, y, test=True):
    # Input:x -> 3,625,145
    # AveragePooling -> 3,312,72
    h = F.average_pooling(x, (2,2), (2,2))
    # Convolution -> 17,312,72
    h = PF.convolution(h, 17, (3,3), (1,1), name='Convolution')
    # BatchNormalization
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
    # AveragePooling_2 -> 17,312,36
    h = F.average_pooling(h, (1,2), (1,2))
    # Convolution_2 -> 60,312,36
    h = PF.convolution(h, 60, (3,3), (1,1), name='Convolution_2')
    # BatchNormalization_2
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # ELU
    h = F.elu(h, 1.0)
    # AveragePooling_3 -> 60,156,18
    h = F.average_pooling(h, (2,2), (2,2))
    # Affine -> 2
    h = PF.affine(h, (2,), name='Affine')
    # Softmax
    h = F.softmax(h)
    # CategoricalCrossEntropy -> 1
    #h = F.categorical_cross_entropy(h, y)

    return h

def network3(x, y, test=True):
    # Input:x -> 3,625,145
    # AveragePooling -> 3,312,72
    h = F.average_pooling(x, (2,2), (2,2))
    # Convolution -> 17,312,72
    h = PF.convolution(h, 17, (3,3), (1,1), name='Convolution')
    # BatchNormalization
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
    # AveragePooling_2 -> 17,156,18
    h = F.average_pooling(h, (2,4), (2,4))
    # Convolution_2 -> 60,156,18
    h = PF.convolution(h, 60, (3,3), (1,1), name='Convolution_2')
    # BatchNormalization_2
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # ELU
    h = F.elu(h, 1.0)
    # AveragePooling_3 -> 60,78,9
    h = F.average_pooling(h, (2,2), (2,2))
    # Affine -> 2
    h = PF.affine(h, (2,), name='Affine')
    # Softmax
    h = F.softmax(h)
    # CategoricalCrossEntropy -> 1
    #h = F.categorical_cross_entropy(h, y)
    return h

def network4(x, y, test=False):
    # Input:x -> 1,50,100
    # MulScalar
    h = F.mul_scalar(x, 0.003921568627451)
    # Convolution_2 -> 32,50,100
    h = PF.convolution(h, 32, (3,3), (1,1), name='Convolution_2')
    # Swish
    h = F.swish(h)
    # BatchNormalization
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
    # MaxPooling -> 32,25,33
    h = F.max_pooling(h, (2,3), (2,3))

    # DepthwiseConvolution
    h1 = PF.depthwise_convolution(h, (3,3), (1,1), name='DepthwiseConvolution')

    # DepthwiseConvolution_2
    h2 = PF.depthwise_convolution(h, (3,3), (1,1), name='DepthwiseConvolution_2')
    # BatchNormalization_3
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # BatchNormalization_2
    h2 = PF.batch_normalization(h2, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # Swish_2
    h1 = F.swish(h1)
    # Swish_4
    h2 = F.swish(h2)

    # Mul2 -> 32,25,33
    h1 = F.mul2(h1, h2)
    # Convolution -> 4,25,33
    h1 = PF.convolution(h1, 4, (3,3), (1,1), name='Convolution')
    # Swish_3
    h1 = F.swish(h1)
    # Affine -> 2
    h1 = PF.affine(h1, (2,), name='Affine')
    # Softmax
    h1 = F.softmax(h1)
    # CategoricalCrossEntropy -> 1
    #h1 = F.categorical_cross_entropy(h1, y)
    return h1

def network5(x, y, test=False):
    # Input:x -> 1,50,100
    # MulScalar
    h = F.mul_scalar(x, 0.003921568627451)

    # ImageAugmentation
    if not test:
        h = F.image_augmentation(h, (1,50,100), (0,0), 1.0, 1.0, 0, 1.0, 0.0, False, False, 0.05, False, 1.0, 0.5, False, 0)

    # Convolution_2 -> 32,50,100
    h = PF.convolution(h, 32, (3,3), (1,1), name='Convolution_2')
    # Swish
    h = F.swish(h)
    # BatchNormalization
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
    # MaxPooling -> 32,12,20
    h = F.max_pooling(h, (4,5), (4,5))
    # Convolution_3 -> 20,12,20
    h = PF.convolution(h, 20, (3,3), (1,1), name='Convolution_3')
    # BatchNormalization_3
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # Swish_2
    h = F.swish(h)
    # Convolution -> 4,12,20
    h = PF.convolution(h, 4, (3,3), (1,1), name='Convolution')
    # Swish_3
    h = F.swish(h)
    # Affine -> 2
    h = PF.affine(h, (2,), name='Affine')
    # Softmax
    h = F.softmax(h)
    # CategoricalCrossEntropy -> 1
    #h = F.categorical_cross_entropy(h, y)
    return h

def network6(x, y, test=False):
    # Input:x -> 1,50,100
    # MulScalar
    h = F.mul_scalar(x, 0.003921568627451)

    # ImageAugmentation
    if not test:
        h = F.image_augmentation(h, (1,50,100), (0,0), 1.0, 1.0, 0, 1.0, 0.0, False, False, 0.05, False, 1.0, 0.5, False, 0)

    # Convolution_2 -> 32,50,100
    h = PF.convolution(h, 32, (1,3), (0,1), name='Convolution_2')
    # Swish
    h = F.swish(h)
    # BatchNormalization
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
    # MaxPooling -> 32,6,14
    h = F.max_pooling(h, (8,7), (8,7))
    # Convolution_3 -> 20,6,14
    h = PF.convolution(h, 20, (3,3), (1,1), name='Convolution_3')
    # BatchNormalization_3
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # Swish_2
    h = F.swish(h)
    # Convolution -> 4,6,14
    h = PF.convolution(h, 4, (3,3), (1,1), name='Convolution')
    # Swish_3
    h = F.swish(h)
    # Affine -> 2
    h = PF.affine(h, (2,), name='Affine')
    # Softmax
    h = F.softmax(h)
    # CategoricalCrossEntropy -> 1
    #h = F.categorical_cross_entropy(h, y)
    return h


class AnomalyDetNetwork:
    def create(type=1):
        """
        """
        cur = os.path.dirname(__file__)

        with nn.parameter_scope('anomaly_det' + str(type)):

            if type == 1:
                nn.load_parameters(cur + '/anomaly_det/20200522_075125/results.nnp')
                x = nn.Variable((1, 3, 625, 145))   #1
                y = network(x, 1, False)
            elif type == 2:
                nn.load_parameters(cur + '/anomaly_det/20200522_075016/results.nnp')
                x = nn.Variable((1, 3, 625, 145))   #1
                y = network2(x, 1, False)
            elif type == 3:
                nn.load_parameters(cur + '/anomaly_det/20200522_074928/results.nnp')
                x = nn.Variable((1, 3, 625, 145))   #1
                y = network3(x, 1, False)
            elif type == 4:
                nn.load_parameters(cur + '/anomaly_det/20200527_032450/results.nnp')
                x = nn.Variable((1, 3, 625, 145))   #1
                y = network2(x, 1, False)
            elif type == 5:
                nn.load_parameters(cur + '/anomaly_det/20200527_085337/results.nnp')
                x = nn.Variable((1, 3, 625, 145))   #1
                y = network2(x, 1, False)
            elif type == 6:
                nn.load_parameters(cur + '/anomaly_det/20200617_053420/results.nnp')
                x = nn.Variable((1, 1, 50, 100))
                y = network4(x, 1, False)
            elif type == 7:
                nn.load_parameters(cur + '/anomaly_det/20200617_063638/results.nnp')
                x = nn.Variable((1, 1, 50, 100))
                y = network5(x, 1, False)
            elif type == 8:
                nn.load_parameters(cur + '/anomaly_det/20200630_040759/results.nnp')
                x = nn.Variable((1, 1, 50, 100))
                y = network5(x, 1, False)

        return x, y
