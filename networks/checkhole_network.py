import os

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

def network(x, y, test=False):
    # Input:x -> 1,50,50
    # ImageAugmentation
    if not test:
        h = F.image_augmentation(x, (1,50,50), (0,0), 1.0, 1.0, 0.15, 1.0, 0.0, True, True, 0.01, False, 0.99, 0.5, False, 0.0)
    else:
        h = x
    # MulScalar
    h = F.mul_scalar(h, 0.003921568627451)
    # AveragePooling -> 1,7,7
    h = F.average_pooling(h, (7,7), (7,7))

    # Convolution_5 -> 13,7,7
    h1 = PF.convolution(h, 13, (3,3), (1,1), name='Convolution_5')

    # Convolution_3 -> 13,7,7
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
    # Convolution_2 -> 13,7,7
    h3 = PF.convolution(h3, 13, (3,3), (1,1), name='Convolution_2')
    # ReLU
    h1 = F.relu(h1, True)
    # ELU_2
    h2 = F.elu(h2, 1.0)
    # BatchNormalization_6
    h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6')
    # ELU_3
    h3 = F.elu(h3, 1.0)

    # Add2 -> 13,7,7
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

    # Mul2_2 -> 13,7,7
    h1 = F.mul2(h1, h2)
    # DepthwiseConvolution -> 13,6,6
    h1 = PF.depthwise_convolution(h1, (4,4), (1,1), name='DepthwiseConvolution')
    # BatchNormalization_4
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')
    # SELU
    h1 = F.selu(h1)
    # Convolution
    h1 = PF.convolution(h1, h1.shape[0 + 1], (3,3), (1,1), name='Convolution')
    # Swish_2
    h1 = F.swish(h1)
    # AveragePooling_3 -> 13,6,1
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


class CheckHoleNetwork:
    """
    hogehoge
    """
    def create(type=0):
        """
        hogehoge
        """
        cur = os.path.dirname(__file__)

        with nn.parameter_scope('checkhole'):
            
            params = [
                ['/checkhole/20220128_084549/results.nnp', (1, 1, 50, 50), network],     # 0
                ['/checkhole/20220131_042954/results.nnp', (1, 1, 50, 50), network],     # 1
                ['/checkhole/20220131_044101/results.nnp', (1, 1, 50, 50), network],     # 2
                ['/checkhole/20220131_050404/results.nnp', (1, 1, 50, 50), network],     # 3
                ['/checkhole/20220131_052346/results.nnp', (1, 1, 50, 50), network],     # 4
                
                [],
            ]

            nn.load_parameters(cur + params[type][0])
            x = nn.Variable(params[type][1])
            y = params[type][2](x, 1, test=True)

        return x, y
