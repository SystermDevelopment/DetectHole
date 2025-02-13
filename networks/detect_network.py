import os

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

def network(x, y, test=False):
    # Input:x -> 1,270,480
    # MulScalar
    h = F.mul_scalar(x, 0.003921568627451)

    # AveragePooling -> 1,30,53
    h = F.average_pooling(h, (9,9), (9,9))
    # DepthwiseConvolution_2
    h = PF.depthwise_convolution(h, (1,1), (0,0), name='DepthwiseConvolution_2')
    # BatchNormalization
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
    # SELU
    h = F.selu(h)
    # DepthwiseConvolution
    h = PF.depthwise_convolution(h, (15,5), (7,2), name='DepthwiseConvolution')
    # BatchNormalization_3
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # SELU_3
    h = F.selu(h)
    # Convolution -> 4,30,53
    h = PF.convolution(h, 4, (3,3), (1,1), name='Convolution')
    # ELU
    h = F.elu(h, 1.0)
    # Affine -> 2
    h = PF.affine(h, (2,), name='Affine')
    # Softmax
    h = F.softmax(h)
    # CategoricalCrossEntropy -> 1
    #h = F.categorical_cross_entropy(h, y)
    return h

def network2(x, y, test=False):
    # Input:x -> 1,270,480
    # MulScalar
    h = F.mul_scalar(x, 0.003921568627451)
    # AveragePooling -> 1,30,40
    h = F.average_pooling(h, (9,12), (9,12))
    # Convolution_2 -> 32,30,40
    h = PF.convolution(h, 32, (3,3), (1,1), name='Convolution_2')
    # BatchNormalization
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
    # LeakyReLU
    h = F.leaky_relu(h, 0.1, True)
    # MaxPooling -> 32,6,8
    h = F.max_pooling(h, (5,5), (5,5))
    # Convolution_3 -> 64,6,8
    h = PF.convolution(h, 64, (3,1), (1,0), name='Convolution_3')
    # BatchNormalization_2
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # LeakyReLU_2
    h = F.leaky_relu(h, 0.1, True)
    # Convolution_4 -> 13,6,8
    h = PF.convolution(h, 13, (3,3), (1,1), name='Convolution_4')
    # BatchNormalization_3
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # Swish_3
    h = F.swish(h)
    # AveragePooling_2 -> 13,3,4
    h = F.average_pooling(h, (2,2), (2,2))
    # Affine -> 2
    h = PF.affine(h, (2,), name='Affine')
    # Softmax
    h = F.softmax(h)
    # CategoricalCrossEntropy -> 1
    #h = F.categorical_cross_entropy(h, y)
    return h

def network3(x, y, test=False):
    # Input:x -> 1,270,480
    # MulScalar
    h = F.mul_scalar(x, 0.003921568627451)
    # AveragePooling -> 1,30,40
    h = F.average_pooling(h, (9,12), (9,12))
    # Convolution_2 -> 32,30,40
    h = PF.convolution(h, 32, (3,3), (1,1), name='Convolution_2')
    # LeakyReLU
    h = F.leaky_relu(h, 0.1, True)
    # MaxPooling -> 32,6,8
    h = F.max_pooling(h, (5,5), (5,5))
    # Convolution_3 -> 64,6,8
    h = PF.convolution(h, 64, (3,1), (1,0), name='Convolution_3')
    # BatchNormalization_2
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # LeakyReLU_2
    h = F.leaky_relu(h, 0.1, True)
    # Convolution_4 -> 13,6,8
    h = PF.convolution(h, 13, (3,3), (1,1), name='Convolution_4')
    # BatchNormalization_3
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # Swish_3
    h = F.swish(h)
    # AveragePooling_2 -> 13,3,1
    h = F.average_pooling(h, (2,8), (2,8))
    # Tanh
    h = F.tanh(h)
    # Dropout
    if not test:
        h = F.dropout(h)
    # Swish
    h = F.swish(h)
    # Affine -> 2
    h = PF.affine(h, (2,), name='Affine')
    # Softmax
    h = F.softmax(h)
    # CategoricalCrossEntropy -> 1
    #h = F.categorical_cross_entropy(h, y)
    return h

def network4(x, y, test=False):
    # Input:x -> 1,270,480
    # MulScalar
    h = F.mul_scalar(x, 0.003921568627451)
    # AveragePooling -> 1,30,40
    h = F.average_pooling(h, (9,12), (9,12))

    # Convolution_5 -> 13,30,40
    h1 = PF.convolution(h, 13, (3,3), (1,1), name='Convolution_5')

    # Convolution_3 -> 13,30,40
    h2 = PF.convolution(h, 13, (3,3), (1,1), name='Convolution_3')

    # Dropout_3
    if not test:
        h3 = F.dropout(h)
    else:
        h3 = h
    # BatchNormalization
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
    # BatchNormalization_3
    h2 = PF.batch_normalization(h2, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # Convolution_2 -> 13,30,40
    h3 = PF.convolution(h3, 13, (3,3), (1,1), name='Convolution_2')
    # ReLU
    h1 = F.relu(h1, True)
    # ELU_2
    h2 = F.elu(h2, 1.0)
    # BatchNormalization_6
    h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6')
    # ELU_3
    h3 = F.elu(h3, 1.0)

    # Add2 -> 13,30,40
    h2 = F.add2(h2, h3, True)
    # DepthwiseConvolution_2
    h2 = PF.depthwise_convolution(h2, (5,5), (2,2), name='DepthwiseConvolution_2')
    # PReLU
    h2 = PF.prelu(h2, 1, False, name='PReLU')
    # Dropout
    if not test:
        h2 = F.dropout(h2)
    # DepthwiseConvolution_3
    h2 = PF.depthwise_convolution(h2, (3,3), (1,1), name='DepthwiseConvolution_3')
    # BatchNormalization_2
    h2 = PF.batch_normalization(h2, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # ELU
    h2 = F.elu(h2, 1.0)

    # Mul2_2 -> 13,30,40
    h1 = F.mul2(h1, h2)
    # DepthwiseConvolution -> 13,29,39
    h1 = PF.depthwise_convolution(h1, (4,4), (1,1), name='DepthwiseConvolution')
    # BatchNormalization_4
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')
    # SELU
    h1 = F.selu(h1)
    # Convolution
    h1 = PF.convolution(h1, h1.shape[0 + 1], (3,3), (1,1), name='Convolution')
    # Swish_2
    h1 = F.swish(h1)
    # AveragePooling_3 -> 13,29,7
    h1 = F.average_pooling(h1, (1,5), (1,5))
    # Tanh_2
    h1 = F.tanh(h1)
    # Dropout_2
    if not test:
        h1 = F.dropout(h1)
    # Swish_4
    h1 = F.swish(h1)
    # Affine -> 2
    h1 = PF.affine(h1, (2,), name='Affine')
    # Softmax
    h1 = F.softmax(h1)
    # CategoricalCrossEntropy -> 1
    #h1 = F.categorical_cross_entropy(h1, y)
    return h1

def network5(x, y, test=False):
    # Input:x -> 1,270,480
    # MulScalar
    h = F.mul_scalar(x, 0.003921568627451)
    # AveragePooling -> 1,30,40
    h = F.average_pooling(h, (9,12), (9,12))

    # Convolution_5 -> 13,30,40
    h1 = PF.convolution(h, 13, (3,3), (1,1), name='Convolution_5')

    # Convolution_3 -> 13,30,40
    h2 = PF.convolution(h, 13, (3,3), (1,1), name='Convolution_3')

    # Dropout_3
    if not test:
        h3 = F.dropout(h)
    else:
        h3 = h
    # BatchNormalization
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
    # BatchNormalization_3
    h2 = PF.batch_normalization(h2, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # Convolution_2 -> 13,30,40
    h3 = PF.convolution(h3, 13, (3,3), (1,1), name='Convolution_2')
    # ReLU
    h1 = F.relu(h1, True)
    # ELU_2
    h2 = F.elu(h2, 1.0)
    # BatchNormalization_6
    h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6')
    # ELU_3
    h3 = F.elu(h3, 1.0)

    # Add2 -> 13,30,40
    h2 = F.add2(h2, h3, True)
    # DepthwiseConvolution_2
    h2 = PF.depthwise_convolution(h2, (5,5), (2,2), name='DepthwiseConvolution_2')
    # PReLU
    h2 = PF.prelu(h2, 1, False, name='PReLU')
    # Dropout
    if not test:
        h2 = F.dropout(h2)
    # DepthwiseConvolution_3
    h2 = PF.depthwise_convolution(h2, (3,3), (1,1), name='DepthwiseConvolution_3')
    # BatchNormalization_2
    h2 = PF.batch_normalization(h2, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # ELU
    h2 = F.elu(h2, 1.0)

    # Mul2_2 -> 13,30,40
    h1 = F.mul2(h1, h2)
    # DepthwiseConvolution -> 13,29,39
    h1 = PF.depthwise_convolution(h1, (4,4), (1,1), name='DepthwiseConvolution')
    # BatchNormalization_4
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')
    # SELU
    h1 = F.selu(h1)
    # Convolution
    h1 = PF.convolution(h1, h1.shape[0 + 1], (3,3), (1,1), name='Convolution')
    # Swish_2
    h1 = F.swish(h1)
    # AveragePooling_3 -> 13,29,7
    h1 = F.average_pooling(h1, (1,8), (1,8))
    # Tanh_2
    h1 = F.tanh(h1)
    # Dropout_2
    if not test:
        h1 = F.dropout(h1)
    # Swish_4
    h1 = F.swish(h1)
    # Affine -> 2
    h1 = PF.affine(h1, (2,), name='Affine')
    # Softmax
    h1 = F.softmax(h1)
    # CategoricalCrossEntropy -> 1
    #h1 = F.categorical_cross_entropy(h1, y)
    return h1

class DetectNetwork:
    """
    hogehoge
    """
    def create(type=0):
        """
        hogehoge
        """
        cur = os.path.dirname(__file__)

        with nn.parameter_scope('detect'):
            
            params = [
                ['/detect/20200609_172350/results.nnp', (1, 1, 270, 480), network],     # 0
                ['/detect/20200617_220905/results.nnp', (1, 1, 270, 480), network2],    # 1
                ['/detect/20200618_120703/results.nnp', (1, 1, 270, 480), network3],    #
                ['/detect/20200618_170333/results.nnp', (1, 1, 270, 480), network4],    #
                ['/detect/20200630_035952/results.nnp', (1, 1, 270, 480), network4],
                ['/detect/20210714_094055/results.nnp', (1, 1, 270, 480), network4],    # 5
                ['/detect/20211102_211025/results.nnp', (1, 1, 270, 480), network4],    # 6 汎用性あり
                ['/detect/20211103_020452/results.nnp', (1, 1, 270, 480), network4],
                ['/detect/20211103_110349/results.nnp', (1, 1, 270, 480), network4],
                ['/detect/20211104_110626/results.nnp', (1, 1, 270, 480), network4],    # 9 
                ['/detect/20220127_232501/results.nnp', (1, 1, 270, 480), network5],    # 10
                ['/detect/20220128_011457/results.nnp', (1, 1, 270, 480), network5],    # 11
                ['/detect/20220130_132727/results.nnp', (1, 1, 270, 480), network5],    # 12
                ['/detect/20220130_135051/results.nnp', (1, 1, 270, 480), network5],    # 13 ***
                ['/detect/20220130_224540/results.nnp', (1, 1, 270, 480), network5],    # 14
                ['/detect/20220130_232856/results.nnp', (1, 1, 270, 480), network5],    # 15 exist少
                
                [],
            ]

            nn.load_parameters(cur + params[type][0])
            x = nn.Variable(params[type][1])
            y = params[type][2](x, 1, test=True)

        return x, y

if __name__ == "__main__":
    dn = DetectNetwork()
    x, y = dn.create(13)
