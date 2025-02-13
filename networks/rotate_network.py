import os

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

def network(x, y, test=False):
    # Input:x -> 1,250,250
    # MaxPooling -> 1,27,35
    h = F.max_pooling(x, (9,7), (9,7))
    # Swish
    h = F.swish(h)

    # Convolution -> 18,27,35
    h1 = PF.convolution(h, 18, (3,3), (1,1), name='Convolution')

    # Convolution_3 -> 18,27,35
    h2 = PF.convolution(h, 18, (1,5), (0,2), name='Convolution_3')

    # Convolution_4 -> 18,27,35
    h3 = PF.convolution(h, 18, (3,3), (1,1), name='Convolution_4')
    # ELU
    h1 = F.elu(h1, 1.0)
    # BatchNormalization_5
    h2 = PF.batch_normalization(h2, (1,), 0.9, 0.0001, not test, name='BatchNormalization_5')
    # BatchNormalization_6
    h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6')
    # ELU_3
    h2 = F.elu(h2, 1.0)
    # Swish_3
    h3 = F.swish(h3)

    # Mul2 -> 18,27,35
    h4 = F.mul2(h1, h2)

    # Mul2_2 -> 18,27,35
    h5 = F.mul2(h1, h3)
    # DepthwiseConvolution_3
    h5 = PF.depthwise_convolution(h5, (5,5), (2,2), name='DepthwiseConvolution_3')
    # Convolution_7
    h4 = PF.convolution(h4, h4.shape[0 + 1], (3,3), (1,1), name='Convolution_7')
    # PReLU_2
    h5 = PF.prelu(h5, 1, False, name='PReLU_2')
    # BatchNormalization_9
    h4 = PF.batch_normalization(h4, (1,), 0.9, 0.0001, not test, name='BatchNormalization_9')
    # ELU_6
    h4 = F.elu(h4, 1.0)
    # DepthwiseConvolution
    h4 = PF.depthwise_convolution(h4, (5,5), (2,2), name='DepthwiseConvolution')
    # BatchNormalization_4
    h4 = PF.batch_normalization(h4, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')
    # ELU_2
    h4 = F.elu(h4, 1.0)
    # Convolution_5
    h4 = PF.convolution(h4, h4.shape[0 + 1], (3,3), (1,1), name='Convolution_5')
    # BatchNormalization_8
    h4 = PF.batch_normalization(h4, (1,), 0.9, 0.0001, not test, name='BatchNormalization_8')
    # ReLU_3
    h4 = F.relu(h4, True)
    # DepthwiseConvolution_2
    h4 = PF.depthwise_convolution(h4, (5,5), (2,2), name='DepthwiseConvolution_2')
    # BatchNormalization_2
    h4 = PF.batch_normalization(h4, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # Swish_2
    h4 = F.swish(h4)

    # DepthwiseConvolution_4
    h6 = PF.depthwise_convolution(h4, (5,5), (2,2), name='DepthwiseConvolution_4')

    # DepthwiseConvolution_5
    h7 = PF.depthwise_convolution(h4, (5,5), (2,2), name='DepthwiseConvolution_5')
    # LeakyReLU
    h6 = F.leaky_relu(h6, 0.1, True)
    # BatchNormalization_10
    h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_10')
    # LeakyReLU_2
    h7 = F.leaky_relu(h7, 0.1, True)

    # Add2_3 -> 18,27,35
    h6 = F.add2(h6, h7, False)
    # Convolution_6
    h6 = PF.convolution(h6, h6.shape[0 + 1], (3,3), (1,1), name='Convolution_6')
    # PReLU
    h6 = PF.prelu(h6, 1, False, name='PReLU')

    # Add2 -> 18,27,35
    h6 = F.add2(h6, h5, True)
    # Dropout
    if not test:
        h6 = F.dropout(h6)
    # Affine -> 1
    h6 = PF.affine(h6, (1,), name='Affine')
    # SquaredError
    #h6 = F.squared_error(h6, y)
    return h6

class RotateNetwork:
    def create():
        """
        """
        cur = os.path.dirname(__file__)

        with nn.parameter_scope('rotate'):
            nn.load_parameters(cur + '/rotate/20200601_081402/results.nnp')     # 05/27
            x = nn.Variable((1, 1, 250, 250))   #6
            y = network(x, 1, test=True)

        return x, y
