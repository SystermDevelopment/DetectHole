import os
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

def network(x, y, test=False):
    # Input:x -> 3,270,480
    # MulScalar
    h = F.mul_scalar(x, 0.003921568627451)
    # MaxPooling -> 3,270,14
    h = F.max_pooling(h, (1,33), (1,33))

    # DepthwiseConvolution_4
    h1 = PF.depthwise_convolution(h, (3,3), (1,1), name='DepthwiseConvolution_4')

    # ReLU_5
    h2 = F.relu(h, False)

    # ReLU_6
    h3 = F.relu(h, False)
    # BatchNormalization
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
    # Convolution
    h3 = PF.convolution(h3, h3.shape[0 + 1], (3,3), (1,1), name='Convolution')
    # Convolution_4
    h2 = PF.convolution(h2, h2.shape[0 + 1], (3,3), (1,1), name='Convolution_4')
    # Swish
    h1 = F.swish(h1)
    # BatchNormalization_3
    h2 = PF.batch_normalization(h2, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # BatchNormalization_5
    h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_5')
    # ReLU_4
    h2 = F.relu(h2, True)
    # LeakyReLU
    h3 = F.leaky_relu(h3, 0.1, True)
    # Convolution_5
    h3 = PF.convolution(h3, h3.shape[0 + 1], (3,3), (1,1), name='Convolution_5')
    # BatchNormalization_7
    h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_7')
    # ReLU
    h3 = F.relu(h3, True)

    # Mul2_2 -> 3,270,14
    h3 = F.mul2(h3, h2)
    # Convolution_3
    h3 = PF.convolution(h3, h3.shape[0 + 1], (5,5), (2,2), name='Convolution_3')
    # BatchNormalization_2
    h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # Swish_2
    h3 = F.swish(h3)

    # Concatenate -> 6,270,14
    h1 = F.concatenate(h1, h3, axis=1)
    # DepthwiseConvolution -> 6,269,13
    h1 = PF.depthwise_convolution(h1, (4,14), (1,6), name='DepthwiseConvolution')
    # ELU_3
    h1 = F.elu(h1, 0.1)

    # DepthwiseConvolution_2
    h4 = PF.depthwise_convolution(h1, (5,5), (2,2), name='DepthwiseConvolution_2')

    # DepthwiseConvolution_3
    h5 = PF.depthwise_convolution(h1, (5,5), (2,2), name='DepthwiseConvolution_3')

    # DepthwiseConvolution_6
    h6 = PF.depthwise_convolution(h1, (5,5), (2,2), name='DepthwiseConvolution_6')
    # ELU_2
    h4 = F.elu(h4, 1.0)
    # BatchNormalization_6
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6')
    # BatchNormalization_4
    h6 = PF.batch_normalization(h6, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')
    # Dropout
    if not test:
        h4 = F.dropout(h4, 0.4444213405832851)
    # ELU_4
    h5 = F.elu(h5, 1.0)
    # ELU
    h6 = F.elu(h6, 1.0)
    # DepthwiseConvolution_8
    h5 = PF.depthwise_convolution(h5, (5,5), (2,2), name='DepthwiseConvolution_8')
    # Convolution_2
    h6 = PF.convolution(h6, h6.shape[0 + 1], (3,3), (1,1), name='Convolution_2')
    # ReLU_3
    h5 = F.relu(h5, True)
    # ELU_6
    h6 = F.elu(h6, 1.0)

    # Add2 -> 6,269,13
    h5 = F.add2(h5, h6, False)
    # DepthwiseConvolution_5
    h5 = PF.depthwise_convolution(h5, (5,5), (2,2), name='DepthwiseConvolution_5')
    # ReLU_2
    h5 = F.relu(h5, True)
    # Dropout_2
    if not test:
        h5 = F.dropout(h5, 0.524858341811534)

    # Mul2 -> 6,269,13
    h4 = F.mul2(h4, h5)
    # Affine -> 155
    h4 = PF.affine(h4, (155,), name='Affine')
    # Tanh
    h4 = F.tanh(h4)
    # Affine_2 -> 4
    h4 = PF.affine(h4, (4,), name='Affine_2')
    # SquaredError
    #h4 = F.squared_error(h4, y)
    return h4



class GetposNetwork:
    def create():
        cur = os.path.dirname(__file__)

        with nn.parameter_scope('getpos'):
            nn.load_parameters(cur + '/getpos/20200531_015845/results.nnp')     # flip版 network 0527撮影データ

            x = nn.Variable((1, 3, 270, 480))
            y = network(x, 1, True)

        return x, y


if __name__ == "__main__":
    getpos = GetposNetwork()
    #getpos.test()
    x, y = GetposNetwork.create()
