import os

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

def network(x, y, test=False):
    # Input:x -> 1,500,500
    # MulScalar
    h = F.mul_scalar(x, 0.003921568627451)
    # MaxPooling -> 1,62,62
    h = F.max_pooling(h, (8,8), (8,8))

    # Convolution_2 -> 6,62,62
    h1 = PF.convolution(h, 6, (3,3), (1,1), name='Convolution_2')

    # Convolution_8
    h2 = PF.convolution(h, h.shape[0 + 1], (5,1), (2,0), name='Convolution_8')
    # BatchNormalization_4
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')
    # BatchNormalization
    h2 = PF.batch_normalization(h2, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
    # Swish_4
    h1 = F.swish(h1)
    # SELU
    h2 = F.selu(h2)
    # Convolution_7
    h2 = PF.convolution(h2, h2.shape[0 + 1], (3,3), (1,1), name='Convolution_7')
    # BatchNormalization_8
    h2 = PF.batch_normalization(h2, (1,), 0.9, 0.0001, not test, name='BatchNormalization_8')
    # LeakyReLU
    h2 = F.leaky_relu(h2, 0.1, True)
    # Convolution_6
    h2 = PF.convolution(h2, h2.shape[0 + 1], (3,3), (1,1), name='Convolution_6')
    # BatchNormalization_3
    h2 = PF.batch_normalization(h2, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # ReLU_2
    h2 = F.relu(h2, True)

    # Add2 -> 6,62,62
    h2 = F.add2(h2, h1, False)
    # Swish_2
    h2 = F.swish(h2)
    # Convolution
    h2 = PF.convolution(h2, h2.shape[0 + 1], (3,3), (1,1), name='Convolution')
    # BatchNormalization_10
    h2 = PF.batch_normalization(h2, (1,), 0.9, 0.0001, not test, name='BatchNormalization_10')
    # ReLU
    h2 = F.relu(h2, True)
    # DepthwiseConvolution_4 -> 6,61,61
    h2 = PF.depthwise_convolution(h2, (4,2), (1,0), name='DepthwiseConvolution_4')
    # BatchNormalization_6
    h2 = PF.batch_normalization(h2, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6')
    # Dropout_2
    if not test:
        h2 = F.dropout(h2, 0.3922920489483754)

    # Convolution_3 -> 4,61,61
    h3 = PF.convolution(h2, 4, (3,3), (1,1), name='Convolution_3')

    # Convolution_5 -> 4,61,61
    h4 = PF.convolution(h2, 4, (3,3), (1,1), name='Convolution_5')
    # BatchNormalization_2
    h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # BatchNormalization_9
    h4 = PF.batch_normalization(h4, (1,), 0.9, 0.0001, not test, name='BatchNormalization_9')
    # Tanh
    h3 = F.tanh(h3)
    # Tanh_2
    h4 = F.tanh(h4)

    # DepthwiseConvolution_2 -> 4,60,60
    h5 = PF.depthwise_convolution(h3, (4,12), (1,5), name='DepthwiseConvolution_2')

    # Concatenate -> 8,61,61
    h6 = F.concatenate(h3, h4, axis=1)
    # DepthwiseConvolution -> 8,60,60
    h6 = PF.depthwise_convolution(h6, (4,12), (1,5), name='DepthwiseConvolution')
    # PReLU_2
    h5 = PF.prelu(h5, 1, False, name='PReLU_2')
    # PReLU
    h6 = PF.prelu(h6, 1, False, name='PReLU')

    # Concatenate_3 -> 12,60,60
    h6 = F.concatenate(h6, h5, axis=1)
    # DepthwiseConvolution_3
    h6 = PF.depthwise_convolution(h6, (7,7), (3,3), name='DepthwiseConvolution_3')
    # BatchNormalization_7
    h6 = PF.batch_normalization(h6, (1,), 0.9, 0.0001, not test, name='BatchNormalization_7')
    # LeakyReLU_2
    h6 = F.leaky_relu(h6, 0.1, True)
    # Dropout_3
    if not test:
        h6 = F.dropout(h6)
    # Convolution_4
    h6 = PF.convolution(h6, h6.shape[0 + 1], (3,3), (1,1), name='Convolution_4')
    # Dropout
    if not test:
        h6 = F.dropout(h6)
    # Affine_2 -> 4
    h6 = PF.affine(h6, (4,), name='Affine_2')
    # BatchNormalization_5
    h6 = PF.batch_normalization(h6, (1,), 0.9, 0.0001, not test, name='BatchNormalization_5')
    # SquaredError
    #h6 = F.squared_error(h6, y)
    return h6


def network9(x, y, test=False):
    # InputX:x -> 1,500,500

    # MulScalar
    h = F.mul_scalar(x, 0.003921568627451)

    # MaxPooling_2 -> 1,100,100
    h = F.max_pooling(h, (5,5), (5,5), False)

    # Convolution -> 64,50,50
    h = PF.convolution(h, 64, (7,7), (3,3), (2,2), with_bias=False, name='Convolution')
    # BatchNormalization
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
    # ReLU
    h = F.relu(h, True)
    # MaxPooling -> 64,25,25
    h = F.max_pooling(h, (3,3), (2,2), True, (1,1))

    # Convolution_3
    h1 = PF.convolution(h, 64, (3,3), (1,1), with_bias=False, name='Convolution_3')
    # BatchNormalization_3
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # ReLU_2
    h1 = F.relu(h1, True)
    # Convolution_4
    h1 = PF.convolution(h1, 64, (3,3), (1,1), with_bias=False, name='Convolution_4')
    # BatchNormalization_4
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')

    # Add2 -> 64,25,25
    h2 = F.add2(h, h1, False)
    # ReLU_4
    h2 = F.relu(h2, True)

    # RepeatStart
    for i in range(1):

        # Convolution_6
        h3 = PF.convolution(h2, 64, (3,3), (1,1), with_bias=False, name='Convolution_6[' + str(i) + ']')
        # BatchNormalization_6
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6[' + str(i) + ']')
        # ReLU_5
        h3 = F.relu(h3, True)
        # Convolution_7
        h3 = PF.convolution(h3, 64, (3,3), (1,1), with_bias=False, name='Convolution_7[' + str(i) + ']')
        # BatchNormalization_7
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_7[' + str(i) + ']')

        # Add2_2 -> 64,25,25
        h4 = F.add2(h2, h3, False)
        # ReLU_7
        h4 = F.relu(h4, True)

        # RepeatEnd
        h2 = h4

    # Convolution_10 -> 128,13,13
    h5 = PF.convolution(h4, 128, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_10')

    # Convolution_9 -> 128,13,13
    h6 = PF.convolution(h4, 128, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_9')
    # BatchNormalization_10
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_10')
    # BatchNormalization_9
    h6 = PF.batch_normalization(h6, (1,), 0.9, 0.0001, not test, name='BatchNormalization_9')
    # ReLU_8
    h5 = F.relu(h5, True)
    # Convolution_11
    h5 = PF.convolution(h5, 128, (3,3), (1,1), with_bias=False, name='Convolution_11')
    # BatchNormalization_11
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_11')

    # Add2_3 -> 128,13,13
    h6 = F.add2(h6, h5, True)
    # ReLU_10
    h6 = F.relu(h6, True)

    # RepeatStart_2
    for i in range(1):

        # Convolution_13
        h7 = PF.convolution(h6, 128, (3,3), (1,1), with_bias=False, name='Convolution_13[' + str(i) + ']')
        # BatchNormalization_13
        h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_13[' + str(i) + ']')
        # ReLU_11
        h7 = F.relu(h7, True)
        # Convolution_14
        h7 = PF.convolution(h7, 128, (3,3), (1,1), with_bias=False, name='Convolution_14[' + str(i) + ']')
        # BatchNormalization_14
        h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_14[' + str(i) + ']')

        # Add2_4 -> 128,13,13
        h8 = F.add2(h6, h7, False)
        # ReLU_13
        h8 = F.relu(h8, True)

        # RepeatEnd_2
        h6 = h8

    # Convolution_17 -> 256,7,7
    h9 = PF.convolution(h8, 256, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_17')

    # Convolution_16 -> 256,7,7
    h10 = PF.convolution(h8, 256, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_16')
    # BatchNormalization_17
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_17')
    # BatchNormalization_16
    h10 = PF.batch_normalization(h10, (1,), 0.9, 0.0001, not test, name='BatchNormalization_16')
    # ReLU_14
    h9 = F.relu(h9, True)
    # Convolution_18
    h9 = PF.convolution(h9, 256, (3,3), (1,1), with_bias=False, name='Convolution_18')
    # BatchNormalization_18
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_18')

    # Add2_5 -> 256,7,7
    h10 = F.add2(h10, h9, True)
    # ReLU_16
    h10 = F.relu(h10, True)

    # RepeatStart_3
    for i in range(1):

        # Convolution_20
        h11 = PF.convolution(h10, 256, (3,3), (1,1), with_bias=False, name='Convolution_20[' + str(i) + ']')
        # BatchNormalization_20
        h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_20[' + str(i) + ']')
        # ReLU_17
        h11 = F.relu(h11, True)
        # Convolution_21
        h11 = PF.convolution(h11, 256, (3,3), (1,1), with_bias=False, name='Convolution_21[' + str(i) + ']')
        # BatchNormalization_21
        h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_21[' + str(i) + ']')

        # Add2_6 -> 256,7,7
        h12 = F.add2(h10, h11, False)
        # ReLU_19
        h12 = F.relu(h12, True)

        # RepeatEnd_3
        h10 = h12

    # Convolution_24 -> 512,4,4
    h13 = PF.convolution(h12, 512, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_24')

    # Convolution_23 -> 512,4,4
    h14 = PF.convolution(h12, 512, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_23')
    # BatchNormalization_24
    h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_24')
    # BatchNormalization_23
    h14 = PF.batch_normalization(h14, (1,), 0.9, 0.0001, not test, name='BatchNormalization_23')
    # ReLU_20
    h13 = F.relu(h13, True)
    # Convolution_25
    h13 = PF.convolution(h13, 512, (3,3), (1,1), with_bias=False, name='Convolution_25')
    # BatchNormalization_25
    h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_25')

    # Add2_8 -> 512,4,4
    h14 = F.add2(h14, h13, True)
    # ReLU_22
    h14 = F.relu(h14, True)

    # RepeatStart_4
    for i in range(1):

        # Convolution_27
        h15 = PF.convolution(h14, 512, (3,3), (1,1), with_bias=False, name='Convolution_27[' + str(i) + ']')
        # BatchNormalization_27
        h15 = PF.batch_normalization(h15, (1,), 0.9, 0.0001, not test, name='BatchNormalization_27[' + str(i) + ']')
        # ReLU_23
        h15 = F.relu(h15, True)
        # Convolution_28
        h15 = PF.convolution(h15, 512, (3,3), (1,1), with_bias=False, name='Convolution_28[' + str(i) + ']')
        # BatchNormalization_28
        h15 = PF.batch_normalization(h15, (1,), 0.9, 0.0001, not test, name='BatchNormalization_28[' + str(i) + ']')

        # Add2_7 -> 512,4,4
        h16 = F.add2(h14, h15, False)
        # ReLU_25
        h16 = F.relu(h16, True)

        # RepeatEnd_4
        h14 = h16

    # AveragePooling -> 512,1,1
    h16 = F.average_pooling(h16, (7,7), (7,7), False)
    # Affine -> 4
    h16 = PF.affine(h16, (4,), name='Affine')

    # SquaredError
    #h16 = F.squared_error(h16, y)
    return h16

def network10(x, y, test=False):
    # InputX:x -> 1,500,500

    # MulScalar
    h = F.mul_scalar(x, 0.003921568627451)

    # MaxPooling_2 -> 1,84,84
    h = F.max_pooling(h, (6,6), (6,6), False)

    # Convolution -> 64,42,42
    h = PF.convolution(h, 64, (7,7), (3,3), (2,2), with_bias=False, name='Convolution')
    # BatchNormalization
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
    # ReLU
    h = F.relu(h, True)
    # MaxPooling -> 64,21,21
    h = F.max_pooling(h, (3,3), (2,2), True, (1,1))

    # Convolution_3
    h1 = PF.convolution(h, 64, (3,3), (1,1), with_bias=False, name='Convolution_3')
    # BatchNormalization_3
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # ReLU_2
    h1 = F.relu(h1, True)
    # Convolution_4
    h1 = PF.convolution(h1, 64, (3,3), (1,1), with_bias=False, name='Convolution_4')
    # BatchNormalization_4
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')

    # Add2 -> 64,21,21
    h2 = F.add2(h, h1, False)
    # ReLU_4
    h2 = F.relu(h2, True)

    # RepeatStart
    for i in range(1):

        # Convolution_6
        h3 = PF.convolution(h2, 64, (3,3), (1,1), with_bias=False, name='Convolution_6[' + str(i) + ']')
        # BatchNormalization_6
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6[' + str(i) + ']')
        # ReLU_5
        h3 = F.relu(h3, True)
        # Convolution_7
        h3 = PF.convolution(h3, 64, (3,3), (1,1), with_bias=False, name='Convolution_7[' + str(i) + ']')
        # BatchNormalization_7
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_7[' + str(i) + ']')

        # Add2_2 -> 64,21,21
        h4 = F.add2(h2, h3, False)
        # ReLU_7
        h4 = F.relu(h4, True)

        # RepeatEnd
        h2 = h4

    # Convolution_10 -> 128,11,11
    h5 = PF.convolution(h4, 128, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_10')

    # Convolution_9 -> 128,11,11
    h6 = PF.convolution(h4, 128, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_9')
    # BatchNormalization_10
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_10')
    # BatchNormalization_9
    h6 = PF.batch_normalization(h6, (1,), 0.9, 0.0001, not test, name='BatchNormalization_9')
    # ReLU_8
    h5 = F.relu(h5, True)
    # Convolution_11
    h5 = PF.convolution(h5, 128, (3,3), (1,1), with_bias=False, name='Convolution_11')
    # BatchNormalization_11
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_11')

    # Add2_3 -> 128,11,11
    h6 = F.add2(h6, h5, True)
    # ReLU_10
    h6 = F.relu(h6, True)

    # RepeatStart_2
    for i in range(1):

        # Convolution_13
        h7 = PF.convolution(h6, 128, (3,3), (1,1), with_bias=False, name='Convolution_13[' + str(i) + ']')
        # BatchNormalization_13
        h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_13[' + str(i) + ']')
        # ReLU_11
        h7 = F.relu(h7, True)
        # Convolution_14
        h7 = PF.convolution(h7, 128, (3,3), (1,1), with_bias=False, name='Convolution_14[' + str(i) + ']')
        # BatchNormalization_14
        h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_14[' + str(i) + ']')

        # Add2_4 -> 128,11,11
        h8 = F.add2(h6, h7, False)
        # ReLU_13
        h8 = F.relu(h8, True)

        # RepeatEnd_2
        h6 = h8

    # Convolution_17 -> 256,6,6
    h9 = PF.convolution(h8, 256, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_17')

    # Convolution_16 -> 256,6,6
    h10 = PF.convolution(h8, 256, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_16')
    # BatchNormalization_17
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_17')
    # BatchNormalization_16
    h10 = PF.batch_normalization(h10, (1,), 0.9, 0.0001, not test, name='BatchNormalization_16')
    # ReLU_14
    h9 = F.relu(h9, True)
    # Convolution_18
    h9 = PF.convolution(h9, 256, (3,3), (1,1), with_bias=False, name='Convolution_18')
    # BatchNormalization_18
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_18')

    # Add2_5 -> 256,6,6
    h10 = F.add2(h10, h9, True)
    # ReLU_16
    h10 = F.relu(h10, True)

    # RepeatStart_3
    for i in range(1):

        # Convolution_20
        h11 = PF.convolution(h10, 256, (3,3), (1,1), with_bias=False, name='Convolution_20[' + str(i) + ']')
        # BatchNormalization_20
        h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_20[' + str(i) + ']')
        # ReLU_17
        h11 = F.relu(h11, True)
        # Convolution_21
        h11 = PF.convolution(h11, 256, (3,3), (1,1), with_bias=False, name='Convolution_21[' + str(i) + ']')
        # BatchNormalization_21
        h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_21[' + str(i) + ']')

        # Add2_6 -> 256,6,6
        h12 = F.add2(h10, h11, False)
        # ReLU_19
        h12 = F.relu(h12, True)

        # RepeatEnd_3
        h10 = h12

    # Convolution_24 -> 512,3,3
    h13 = PF.convolution(h12, 512, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_24')

    # Convolution_23 -> 512,3,3
    h14 = PF.convolution(h12, 512, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_23')
    # BatchNormalization_24
    h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_24')
    # BatchNormalization_23
    h14 = PF.batch_normalization(h14, (1,), 0.9, 0.0001, not test, name='BatchNormalization_23')
    # ReLU_20
    h13 = F.relu(h13, True)
    # Convolution_25
    h13 = PF.convolution(h13, 512, (3,3), (1,1), with_bias=False, name='Convolution_25')
    # BatchNormalization_25
    h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_25')

    # Add2_8 -> 512,3,3
    h14 = F.add2(h14, h13, True)
    # ReLU_22
    h14 = F.relu(h14, True)

    # RepeatStart_4
    for i in range(1):

        # Convolution_27
        h15 = PF.convolution(h14, 512, (3,3), (1,1), with_bias=False, name='Convolution_27[' + str(i) + ']')
        # BatchNormalization_27
        h15 = PF.batch_normalization(h15, (1,), 0.9, 0.0001, not test, name='BatchNormalization_27[' + str(i) + ']')
        # ReLU_23
        h15 = F.relu(h15, True)
        # Convolution_28
        h15 = PF.convolution(h15, 512, (3,3), (1,1), with_bias=False, name='Convolution_28[' + str(i) + ']')
        # BatchNormalization_28
        h15 = PF.batch_normalization(h15, (1,), 0.9, 0.0001, not test, name='BatchNormalization_28[' + str(i) + ']')

        # Add2_7 -> 512,3,3
        h16 = F.add2(h14, h15, False)
        # ReLU_25
        h16 = F.relu(h16, True)

        # RepeatEnd_4
        h14 = h16

    # Affine -> 4
    h16 = PF.affine(h16, (4,), name='Affine')

    # SquaredError
    #h16 = F.squared_error(h16, y)
    return h16

def network11(x, y, test=False):
    # InputX:x -> 1,500,500

    # MulScalar
    h = F.mul_scalar(x, 0.003921568627451)

    # MaxPooling_2 -> 1,72,63
    h = F.max_pooling(h, (7,8), (7,8), False)

    # Convolution -> 64,36,32
    h = PF.convolution(h, 64, (7,7), (3,3), (2,2), with_bias=False, name='Convolution')
    # BatchNormalization
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
    # ReLU
    h = F.relu(h, True)
    # MaxPooling -> 64,18,16
    h = F.max_pooling(h, (3,3), (2,2), True, (1,1))

    # Convolution_3
    h1 = PF.convolution(h, 64, (3,3), (1,1), with_bias=False, name='Convolution_3')
    # BatchNormalization_3
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # ReLU_2
    h1 = F.relu(h1, True)
    # Convolution_4
    h1 = PF.convolution(h1, 64, (3,3), (1,1), with_bias=False, name='Convolution_4')
    # BatchNormalization_4
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')

    # Add2 -> 64,18,16
    h2 = F.add2(h, h1, False)
    # ReLU_4
    h2 = F.relu(h2, True)

    # RepeatStart
    for i in range(1):

        # Convolution_6
        h3 = PF.convolution(h2, 64, (3,3), (1,1), with_bias=False, name='Convolution_6[' + str(i) + ']')
        # BatchNormalization_6
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6[' + str(i) + ']')
        # ReLU_5
        h3 = F.relu(h3, True)
        # Convolution_7
        h3 = PF.convolution(h3, 64, (3,3), (1,1), with_bias=False, name='Convolution_7[' + str(i) + ']')
        # BatchNormalization_7
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_7[' + str(i) + ']')

        # Add2_2 -> 64,18,16
        h4 = F.add2(h2, h3, False)
        # ReLU_7
        h4 = F.relu(h4, True)

        # RepeatEnd
        h2 = h4

    # Convolution_10 -> 128,9,8
    h5 = PF.convolution(h4, 128, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_10')

    # Convolution_9 -> 128,9,8
    h6 = PF.convolution(h4, 128, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_9')
    # BatchNormalization_10
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_10')
    # BatchNormalization_9
    h6 = PF.batch_normalization(h6, (1,), 0.9, 0.0001, not test, name='BatchNormalization_9')
    # ReLU_8
    h5 = F.relu(h5, True)
    # Convolution_11
    h5 = PF.convolution(h5, 128, (3,3), (1,1), with_bias=False, name='Convolution_11')
    # BatchNormalization_11
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_11')

    # Add2_3 -> 128,9,8
    h6 = F.add2(h6, h5, True)
    # ReLU_10
    h6 = F.relu(h6, True)

    # RepeatStart_2
    for i in range(1):

        # Convolution_13
        h7 = PF.convolution(h6, 128, (3,3), (1,1), with_bias=False, name='Convolution_13[' + str(i) + ']')
        # BatchNormalization_13
        h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_13[' + str(i) + ']')
        # ReLU_11
        h7 = F.relu(h7, True)
        # Convolution_14
        h7 = PF.convolution(h7, 128, (3,3), (1,1), with_bias=False, name='Convolution_14[' + str(i) + ']')
        # BatchNormalization_14
        h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_14[' + str(i) + ']')

        # Add2_4 -> 128,9,8
        h8 = F.add2(h6, h7, False)
        # ReLU_13
        h8 = F.relu(h8, True)

        # RepeatEnd_2
        h6 = h8

    # Convolution_17 -> 256,5,4
    h9 = PF.convolution(h8, 256, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_17')

    # Convolution_16 -> 256,5,4
    h10 = PF.convolution(h8, 256, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_16')
    # BatchNormalization_17
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_17')
    # BatchNormalization_16
    h10 = PF.batch_normalization(h10, (1,), 0.9, 0.0001, not test, name='BatchNormalization_16')
    # ReLU_14
    h9 = F.relu(h9, True)
    # Convolution_18
    h9 = PF.convolution(h9, 256, (3,3), (1,1), with_bias=False, name='Convolution_18')
    # BatchNormalization_18
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_18')

    # Add2_5 -> 256,5,4
    h10 = F.add2(h10, h9, True)
    # ReLU_16
    h10 = F.relu(h10, True)

    # RepeatStart_3
    for i in range(1):

        # Convolution_20
        h11 = PF.convolution(h10, 256, (3,3), (1,1), with_bias=False, name='Convolution_20[' + str(i) + ']')
        # BatchNormalization_20
        h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_20[' + str(i) + ']')
        # ReLU_17
        h11 = F.relu(h11, True)
        # Convolution_21
        h11 = PF.convolution(h11, 256, (3,3), (1,1), with_bias=False, name='Convolution_21[' + str(i) + ']')
        # BatchNormalization_21
        h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_21[' + str(i) + ']')

        # Add2_6 -> 256,5,4
        h12 = F.add2(h10, h11, False)
        # ReLU_19
        h12 = F.relu(h12, True)

        # RepeatEnd_3
        h10 = h12

    # Convolution_24 -> 512,3,2
    h13 = PF.convolution(h12, 512, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_24')

    # Convolution_23 -> 512,3,2
    h14 = PF.convolution(h12, 512, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_23')
    # BatchNormalization_24
    h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_24')
    # BatchNormalization_23
    h14 = PF.batch_normalization(h14, (1,), 0.9, 0.0001, not test, name='BatchNormalization_23')
    # ReLU_20
    h13 = F.relu(h13, True)
    # Convolution_25
    h13 = PF.convolution(h13, 512, (3,3), (1,1), with_bias=False, name='Convolution_25')
    # BatchNormalization_25
    h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_25')

    # Add2_8 -> 512,3,2
    h14 = F.add2(h14, h13, True)
    # ReLU_22
    h14 = F.relu(h14, True)

    # RepeatStart_4
    for i in range(1):

        # Convolution_27
        h15 = PF.convolution(h14, 512, (3,3), (1,1), with_bias=False, name='Convolution_27[' + str(i) + ']')
        # BatchNormalization_27
        h15 = PF.batch_normalization(h15, (1,), 0.9, 0.0001, not test, name='BatchNormalization_27[' + str(i) + ']')
        # ReLU_23
        h15 = F.relu(h15, True)
        # Convolution_28
        h15 = PF.convolution(h15, 512, (3,3), (1,1), with_bias=False, name='Convolution_28[' + str(i) + ']')
        # BatchNormalization_28
        h15 = PF.batch_normalization(h15, (1,), 0.9, 0.0001, not test, name='BatchNormalization_28[' + str(i) + ']')

        # Add2_7 -> 512,3,2
        h16 = F.add2(h14, h15, False)
        # ReLU_25
        h16 = F.relu(h16, True)

        # RepeatEnd_4
        h14 = h16

    # Affine -> 4
    h16 = PF.affine(h16, (4,), name='Affine')

    # SquaredError
    #h16 = F.squared_error(h16, y)
    return h16

def network12(x, y, test=False):
    # InputX:x -> 1,500,500
    # MulScalar
    h = F.mul_scalar(x, 0.003921568627451)
    # MaxPooling_2 -> 1,72,100
    h = F.max_pooling(h, (7,5), (7,5), False)
    # Convolution -> 64,36,50
    h = PF.convolution(h, 64, (7,7), (3,3), (2,2), with_bias=False, name='Convolution')
    # BatchNormalization
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
    # ReLU
    h = F.relu(h, True)
    # MaxPooling -> 64,18,25
    h = F.max_pooling(h, (3,3), (2,2), True, (1,1))

    # Convolution_3
    h1 = PF.convolution(h, 64, (3,3), (1,1), with_bias=False, name='Convolution_3')
    # BatchNormalization_3
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # ReLU_2
    h1 = F.relu(h1, True)
    # Convolution_4
    h1 = PF.convolution(h1, 64, (3,3), (1,1), with_bias=False, name='Convolution_4')
    # BatchNormalization_4
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')

    # Add2 -> 64,18,25
    h2 = F.add2(h, h1, False)
    # ReLU_4
    h2 = F.relu(h2, True)
    # RepeatStart
    for i in range(1):

        # Convolution_6
        h3 = PF.convolution(h2, 64, (3,3), (1,1), with_bias=False, name='Convolution_6[' + str(i) + ']')
        # BatchNormalization_6
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6[' + str(i) + ']')
        # ReLU_5
        h3 = F.relu(h3, True)
        # Convolution_7
        h3 = PF.convolution(h3, 64, (3,3), (1,1), with_bias=False, name='Convolution_7[' + str(i) + ']')
        # BatchNormalization_7
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_7[' + str(i) + ']')

        # Add2_2 -> 64,18,25
        h4 = F.add2(h2, h3, False)
        # ReLU_7
        h4 = F.relu(h4, True)
        # RepeatEnd
        h2 = h4

    # Convolution_10 -> 128,9,13
    h5 = PF.convolution(h4, 128, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_10')

    # Convolution_9 -> 128,9,13
    h6 = PF.convolution(h4, 128, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_9')
    # BatchNormalization_10
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_10')
    # BatchNormalization_9
    h6 = PF.batch_normalization(h6, (1,), 0.9, 0.0001, not test, name='BatchNormalization_9')
    # ReLU_8
    h5 = F.relu(h5, True)
    # Convolution_11
    h5 = PF.convolution(h5, 128, (3,3), (1,1), with_bias=False, name='Convolution_11')
    # BatchNormalization_11
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_11')

    # Add2_3 -> 128,9,13
    h6 = F.add2(h6, h5, True)
    # ReLU_10
    h6 = F.relu(h6, True)
    # RepeatStart_2
    for i in range(1):

        # Convolution_13
        h7 = PF.convolution(h6, 128, (3,3), (1,1), with_bias=False, name='Convolution_13[' + str(i) + ']')
        # BatchNormalization_13
        h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_13[' + str(i) + ']')
        # ReLU_11
        h7 = F.relu(h7, True)
        # Convolution_14
        h7 = PF.convolution(h7, 128, (3,3), (1,1), with_bias=False, name='Convolution_14[' + str(i) + ']')
        # BatchNormalization_14
        h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_14[' + str(i) + ']')

        # Add2_4 -> 128,9,13
        h8 = F.add2(h6, h7, False)
        # ReLU_13
        h8 = F.relu(h8, True)
        # RepeatEnd_2
        h6 = h8

    # Convolution_17 -> 256,5,7
    h9 = PF.convolution(h8, 256, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_17')

    # Convolution_16 -> 256,5,7
    h10 = PF.convolution(h8, 256, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_16')
    # BatchNormalization_17
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_17')
    # BatchNormalization_16
    h10 = PF.batch_normalization(h10, (1,), 0.9, 0.0001, not test, name='BatchNormalization_16')
    # ReLU_14
    h9 = F.relu(h9, True)
    # Convolution_18
    h9 = PF.convolution(h9, 256, (3,3), (1,1), with_bias=False, name='Convolution_18')
    # BatchNormalization_18
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_18')

    # Add2_5 -> 256,5,7
    h10 = F.add2(h10, h9, True)
    # ReLU_16
    h10 = F.relu(h10, True)
    # RepeatStart_3
    for i in range(1):
        # ReLU_19
        h10 = F.relu(h10, False)
        # RepeatEnd_3

    # Convolution_24 -> 512,3,4
    h11 = PF.convolution(h10, 512, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_24')

    # Convolution_23 -> 512,3,4
    h12 = PF.convolution(h10, 512, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_23')
    # BatchNormalization_24
    h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_24')
    # BatchNormalization_23
    h12 = PF.batch_normalization(h12, (1,), 0.9, 0.0001, not test, name='BatchNormalization_23')
    # ReLU_20
    h11 = F.relu(h11, True)
    # Convolution_25
    h11 = PF.convolution(h11, 512, (3,3), (1,1), with_bias=False, name='Convolution_25')
    # BatchNormalization_25
    h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_25')

    # Add2_8 -> 512,3,4
    h12 = F.add2(h12, h11, True)
    # ReLU_22
    h12 = F.relu(h12, True)
    # RepeatStart_4
    for i in range(1):

        # Convolution_27
        h13 = PF.convolution(h12, 512, (3,3), (1,1), with_bias=False, name='Convolution_27[' + str(i) + ']')
        # BatchNormalization_27
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_27[' + str(i) + ']')
        # ReLU_23
        h13 = F.relu(h13, True)
        # Convolution_28
        h13 = PF.convolution(h13, 512, (3,3), (1,1), with_bias=False, name='Convolution_28[' + str(i) + ']')
        # BatchNormalization_28
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_28[' + str(i) + ']')

        # Add2_7 -> 512,3,4
        h14 = F.add2(h12, h13, False)
        # ReLU_25
        h14 = F.relu(h14, True)
        # RepeatEnd_4
        h12 = h14
    # Affine -> 4
    h14 = PF.affine(h14, (4,), name='Affine')
    # SquaredError
    #h14 = F.squared_error(h14, y)
    return h14

class GetHoleNetwork:
    def create(type=1):
        cur = os.path.dirname(__file__)

        with nn.parameter_scope('gethole' + str(type)):
            if type == 1:
                nn.load_parameters(cur + '/gethole/20200625_180801/results.nnp')     # network 0602撮影データ
                x = nn.Variable((1, 1, 500, 500))   # 1,2,3
                y = network(x, 1, True)
            elif type == 9:
                nn.load_parameters(cur + '/gethole/20200827_192236/results.nnp')     # network 070E撮影データ
                x = nn.Variable((1, 1, 500, 500))  # 4,5
                y = network9(x, 1, True)
            elif type == 10:
                nn.load_parameters(cur + '/gethole/20210714_213908/results.nnp')     # network 0602撮影データ
                x = nn.Variable((1, 1, 500, 500))  # 4,5
                y = network9(x, 1, True)
            elif type == 11:
                # ほぼ問題なし、リングが左右端の場合に少しずれる
                nn.load_parameters(cur + '/gethole/20211105_235834/results.nnp')     # network 1027撮影データ ring8
                x = nn.Variable((1, 1, 500, 500))  # 4,5
                y = network10(x, 1, True)
            elif type == 12:
                # 動作確認中
                nn.load_parameters(cur + '/gethole/20211106_040905/results.nnp')     # network 1027撮影データ ring8
                x = nn.Variable((1, 1, 500, 500))  # 4,5
                y = network10(x, 1, True)
            elif type == 13:
                # 動作確認中
                nn.load_parameters(cur + '/gethole/20211106_190553/results.nnp')     # network 1027撮影データ ring8
                x = nn.Variable((1, 1, 500, 500))  # 4,5
                y = network11(x, 1, True)
            elif type == 14:
                # 動作確認中
                nn.load_parameters(cur + '/gethole/20211107_010943/results.nnp')     # network 1027撮影データ ring9
                x = nn.Variable((1, 1, 500, 500))  # 4,5
                y = network12(x, 1, True)
            elif type == 15:
                # 動作確認中
                nn.load_parameters(cur + '/gethole/20220128_040729/results.nnp')     # network 1220撮影データ ring10
                x = nn.Variable((1, 1, 500, 500))  # 4,5
                y = network12(x, 1, True)
            elif type == 16:
                # 動作確認中
                nn.load_parameters(cur + '/gethole/20220131_024832/results.nnp')     # network 1220撮影データ ring10
                x = nn.Variable((1, 1, 500, 500))  # 4,5
                y = network12(x, 1, True)
        return x, y


if __name__ == "__main__":
    getpos = GetHoleNetwork()
    #getpos.test()
    x, y = GetHoleNetwork.create(6)
