import os
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

def network(x, y, test=False):
    # InputX:x -> 1,480,245
    # MulScalar_2
    h = F.mul_scalar(x, 0.003921568627451)
    # ImageAugmentation
    if not test:
        h = F.image_augmentation(h, (1,480,245), (0,0), 1.0, 1.0, 0.1, 1.0, 0.0, False, False, 0.0, False, 1.0, 0.5, False, 0.0)
    # AveragePooling -> 1,80,28
    h = F.average_pooling(h, (6,9), (6,9), False)
    # Convolution_2 -> 64,40,14
    h = PF.convolution(h, 64, (7,7), (3,3), (2,2), with_bias=False, name='Convolution_2')
    # BatchNormalization_2
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # ReLU_2
    h = F.relu(h, True)
    # MaxPooling_2 -> 64,19,7
    h = F.max_pooling(h, (6,4), (2,2), True, (1,1))

    # Convolution_3
    h1 = PF.convolution(h, 64, (1,1), (0,0), with_bias=False, name='Convolution_3')
    # BatchNormalization_3
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # ReLU_4
    h1 = F.relu(h1, True)
    # Convolution_5
    h1 = PF.convolution(h1, 64, (3,3), (1,1), with_bias=False, name='Convolution_5')
    # BatchNormalization_4
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')

    # Add2 -> 64,19,7
    h2 = F.add2(h, h1, False)
    # ReLU_5
    h2 = F.relu(h2, True)
    # RepeatStart_2
    for i in range(2):

        # Convolution_6
        h3 = PF.convolution(h2, 64, (3,3), (1,1), with_bias=False, name='Convolution_6[' + str(i) + ']')
        # BatchNormalization_6
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6[' + str(i) + ']')
        # ReLU_6
        h3 = F.relu(h3, True)
        # Convolution_7
        h3 = PF.convolution(h3, 64, (1,5), (0,2), with_bias=False, name='Convolution_7[' + str(i) + ']')
        # BatchNormalization_7
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_7[' + str(i) + ']')

        # Add2_2 -> 64,19,7
        h4 = F.add2(h2, h3, False)
        # ReLU_7
        h4 = F.relu(h4, True)
        # RepeatEnd_2
        h2 = h4

    # Convolution_10 -> 124,10,4
    h5 = PF.convolution(h4, 124, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_10')

    # Convolution_9 -> 128,10,4
    h6 = PF.convolution(h4, 128, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_9')
    # BatchNormalization_10
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_10')
    # BatchNormalization_9
    h6 = PF.batch_normalization(h6, (1,), 0.9, 0.0001, not test, name='BatchNormalization_9')
    # ReLU_8
    h5 = F.relu(h5, True)
    # Convolution_11 -> 128,10,4
    h5 = PF.convolution(h5, 128, (3,3), (1,1), with_bias=False, name='Convolution_11')
    # BatchNormalization_11
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_11')

    # Add2_3 -> 128,10,4
    h6 = F.add2(h6, h5, True)
    # ReLU_10
    h6 = F.relu(h6, True)
    # RepeatStart_3
    for i in range(2):

        # Convolution_13
        h7 = PF.convolution(h6, 128, (3,3), (1,1), with_bias=False, name='Convolution_13[' + str(i) + ']')
        # BatchNormalization_13
        h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_13[' + str(i) + ']')
        # Convolution_14
        h7 = PF.convolution(h7, 128, (3,3), (1,1), with_bias=False, name='Convolution_14[' + str(i) + ']')
        # BatchNormalization_14
        h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_14[' + str(i) + ']')

        # Add2_4 -> 128,10,4
        h8 = F.add2(h6, h7, False)
        # ReLU_13
        h8 = F.relu(h8, True)
        # RepeatEnd_3
        h6 = h8

    # Convolution_17 -> 256,5,2
    h9 = PF.convolution(h8, 256, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_17')

    # Convolution_16 -> 256,5,2
    h10 = PF.convolution(h8, 256, (3,1), (1,0), (2,2), with_bias=False, name='Convolution_16')
    # BatchNormalization_17
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_17')
    # BatchNormalization_16
    h10 = PF.batch_normalization(h10, (1,), 0.9, 0.0001, not test, name='BatchNormalization_16')
    # Swish
    h9 = F.swish(h9)
    # Convolution_18
    h9 = PF.convolution(h9, 256, (3,3), (1,1), with_bias=False, name='Convolution_18')
    # BatchNormalization_18
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_18')

    # Add2_5 -> 256,5,2
    h10 = F.add2(h10, h9, True)
    # Tanh
    h10 = F.tanh(h10)
    # RepeatStart_4
    for i in range(1):
        # ReLU_19
        h10 = F.relu(h10, False)
        # RepeatEnd_4

    # Convolution_24 -> 512,3,1
    h11 = PF.convolution(h10, 512, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_24')

    # Convolution_23 -> 512,3,1
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

    # Add2_8 -> 512,3,1
    h12 = F.add2(h12, h11, True)
    # ReLU_22
    h12 = F.relu(h12, True)
    # RepeatStart_5
    for i in range(2):

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

        # Add2_7 -> 512,3,1
        h14 = F.add2(h12, h13, False)
        # ReLU_25
        h14 = F.relu(h14, True)
        # RepeatEnd_5
        h12 = h14
    # Affine_2 -> 2
    h14 = PF.affine(h14, (2,), name='Affine_2')
    # SquaredError_2
    #h14 = F.squared_error(h14, y)
    return h14

def network2(x, y, test=False):
    # InputX:x -> 1,480,245
    # MulScalar_2
    h = F.mul_scalar(x, 0.003921568627451)
    # ImageAugmentation
    if not test:
        h = F.image_augmentation(h, (1,480,245), (0,0), 1.0, 1.0, 0.1, 1.0, 0.0, False, False, 0.0, False, 1.0, 0.5, False, 0.0)
    # AveragePooling -> 1,60,49
    h = F.average_pooling(h, (8,5), (8,5), False)
    # Convolution_2 -> 64,30,25
    h = PF.convolution(h, 64, (7,7), (3,3), (2,2), with_bias=False, name='Convolution_2')
    # BatchNormalization_2
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # ReLU_2
    h = F.relu(h, True)
    # AveragePooling_2 -> 64,11,12
    h = F.average_pooling(h, (11,4), (2,2), True, (1,1))

    # Convolution_3
    h1 = PF.convolution(h, 64, (3,3), (1,1), with_bias=False, name='Convolution_3')
    # BatchNormalization_3
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # ReLU_4
    h1 = F.relu(h1, True)
    # Convolution_5
    h1 = PF.convolution(h1, 64, (3,3), (1,1), with_bias=False, name='Convolution_5')
    # BatchNormalization_4
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')

    # Add2 -> 64,11,12
    h2 = F.add2(h, h1, False)
    # ReLU_5
    h2 = F.relu(h2, True)
    # RepeatStart_2
    for i in range(2):

        # Convolution_6 -> 43,11,12
        h3 = PF.convolution(h2, 43, (3,3), (1,1), with_bias=False, name='Convolution_6[' + str(i) + ']')
        # BatchNormalization_6
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6[' + str(i) + ']')
        # ReLU_6
        h3 = F.relu(h3, True)
        # Convolution_7 -> 64,11,12
        h3 = PF.convolution(h3, 64, (3,3), (1,1), with_bias=False, name='Convolution_7[' + str(i) + ']')
        # BatchNormalization_7
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_7[' + str(i) + ']')

        # Add2_2 -> 64,11,12
        h4 = F.add2(h2, h3, False)
        # ReLU_7
        h4 = F.relu(h4, True)
        # RepeatEnd_2
        h2 = h4

    # Convolution_10 -> 128,6,6
    h5 = PF.convolution(h4, 128, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_10')

    # Convolution_9 -> 128,6,6
    h6 = PF.convolution(h4, 128, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_9')
    # BatchNormalization_10
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_10')
    # BatchNormalization_9
    h6 = PF.batch_normalization(h6, (1,), 0.9, 0.0001, not test, name='BatchNormalization_9')
    # ReLU_8
    h5 = F.relu(h5, True)
    # Convolution_11
    h5 = PF.convolution(h5, 128, (5,3), (2,1), with_bias=False, name='Convolution_11')
    # BatchNormalization_11
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_11')

    # Add2_3 -> 128,6,6
    h6 = F.add2(h6, h5, True)
    # ReLU_10
    h6 = F.relu(h6, True)
    # RepeatStart_3
    for i in range(2):

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

        # Add2_4 -> 128,6,6
        h8 = F.add2(h6, h7, False)
        # ReLU_13
        h8 = F.relu(h8, True)
        # RepeatEnd_3
        h6 = h8

    # Convolution_17 -> 256,3,3
    h9 = PF.convolution(h8, 256, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_17')

    # Convolution_16 -> 256,3,3
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

    # Add2_5 -> 256,3,3
    h10 = F.add2(h10, h9, True)
    # ReLU_16
    h10 = F.relu(h10, True)
    # RepeatStart_4
    for i in range(1):
        # ReLU_19
        h10 = F.relu(h10, False)
        # RepeatEnd_4

    # Convolution_24 -> 512,2,2
    h11 = PF.convolution(h10, 512, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_24')

    # Convolution_23 -> 512,2,2
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

    # Add2_8 -> 512,2,2
    h12 = F.add2(h12, h11, True)
    # ReLU_22
    h12 = F.relu(h12, True)
    # RepeatStart_5
    for i in range(1):

        # Convolution_27
        h13 = PF.convolution(h12, 512, (3,3), (1,1), with_bias=False, name='Convolution_27[' + str(i) + ']')
        # BatchNormalization_5
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_5[' + str(i) + ']')
        # PReLU
        h13 = PF.prelu(h13, 1, False, name='PReLU[' + str(i) + ']')
        # DepthwiseConvolution_2
        h13 = PF.depthwise_convolution(h13, (5,5), (2,2), name='DepthwiseConvolution_2[' + str(i) + ']')
        # BatchNormalization
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization[' + str(i) + ']')
        # ReLU
        h13 = F.relu(h13, True)
        # DepthwiseConvolution
        h13 = PF.depthwise_convolution(h13, (5,5), (2,2), name='DepthwiseConvolution[' + str(i) + ']')
        # BatchNormalization_27
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_27[' + str(i) + ']')
        # ReLU_23
        h13 = F.relu(h13, True)
        # Convolution_28
        h13 = PF.convolution(h13, 512, (3,3), (1,1), with_bias=False, name='Convolution_28[' + str(i) + ']')
        # BatchNormalization_28
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_28[' + str(i) + ']')

        # Add2_7 -> 512,2,2
        h14 = F.add2(h12, h13, False)
        # ReLU_25
        h14 = F.relu(h14, True)
        # RepeatEnd_5
        h12 = h14
    # Affine_2 -> 2
    h14 = PF.affine(h14, (2,), name='Affine_2')
    # SquaredError_2
    #h14 = F.squared_error(h14, y)
    return h14

def network3(x, y, test=False):
    # InputX:x -> 1,480,245
    # MulScalar_2
    h = F.mul_scalar(x, 0.003921568627451)
    # ImageAugmentation
    if not test:
        h = F.image_augmentation(h, (1,480,245), (0,0), 1.0, 1.0, 0.1, 1.0, 0.0, False, False, 0.0, False, 1.0, 0.5, False, 0.0)
    # AveragePooling -> 1,60,49
    h = F.average_pooling(h, (8,5), (8,5), False)
    # Convolution_2 -> 64,30,25
    h = PF.convolution(h, 64, (7,7), (3,3), (2,2), with_bias=False, name='Convolution_2')
    # BatchNormalization_2
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # ReLU_2
    h = F.relu(h, True)
    # AveragePooling_2 -> 64,11,12
    h = F.average_pooling(h, (11,4), (2,2), True, (1,1))

    # Convolution_3
    h1 = PF.convolution(h, 64, (3,3), (1,1), with_bias=False, name='Convolution_3')
    # BatchNormalization_3
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # ReLU_4
    h1 = F.relu(h1, True)
    # Convolution_5
    h1 = PF.convolution(h1, 64, (3,3), (1,1), with_bias=False, name='Convolution_5')
    # BatchNormalization_4
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')

    # Add2 -> 64,11,12
    h2 = F.add2(h, h1, False)
    # ReLU_5
    h2 = F.relu(h2, True)
    # RepeatStart_2
    for i in range(2):

        # Convolution_6 -> 43,11,12
        h3 = PF.convolution(h2, 43, (3,3), (1,1), with_bias=False, name='Convolution_6[' + str(i) + ']')
        # BatchNormalization_6
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6[' + str(i) + ']')
        # ReLU_6
        h3 = F.relu(h3, True)
        # Convolution_7 -> 64,11,12
        h3 = PF.convolution(h3, 64, (3,3), (1,1), with_bias=False, name='Convolution_7[' + str(i) + ']')
        # BatchNormalization_7
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_7[' + str(i) + ']')

        # Add2_2 -> 64,11,12
        h4 = F.add2(h2, h3, False)
        # ReLU_7
        h4 = F.relu(h4, True)
        # RepeatEnd_2
        h2 = h4

    # Convolution_10 -> 128,6,6
    h5 = PF.convolution(h4, 128, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_10')

    # Convolution_9 -> 128,6,6
    h6 = PF.convolution(h4, 128, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_9')
    # BatchNormalization_10
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_10')
    # BatchNormalization_9
    h6 = PF.batch_normalization(h6, (1,), 0.9, 0.0001, not test, name='BatchNormalization_9')
    # ReLU_8
    h5 = F.relu(h5, True)
    # Convolution_11
    h5 = PF.convolution(h5, 128, (5,3), (2,1), with_bias=False, name='Convolution_11')
    # BatchNormalization_11
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_11')

    # Add2_3 -> 128,6,6
    h6 = F.add2(h6, h5, True)
    # ReLU_10
    h6 = F.relu(h6, True)
    # RepeatStart_3
    for i in range(2):

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

        # Add2_4 -> 128,6,6
        h8 = F.add2(h6, h7, False)
        # ReLU_13
        h8 = F.relu(h8, True)
        # RepeatEnd_3
        h6 = h8

    # Convolution_17 -> 256,3,3
    h9 = PF.convolution(h8, 256, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_17')

    # Convolution_16 -> 256,3,3
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

    # Add2_5 -> 256,3,3
    h10 = F.add2(h10, h9, True)
    # ReLU_16
    h10 = F.relu(h10, True)
    # RepeatStart_4
    for i in range(1):
        # ReLU_19
        h10 = F.relu(h10, False)
        # RepeatEnd_4

    # Convolution_24 -> 512,2,2
    h11 = PF.convolution(h10, 512, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_24')

    # Convolution_23 -> 512,2,2
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

    # Add2_8 -> 512,2,2
    h12 = F.add2(h12, h11, True)
    # ReLU_22
    h12 = F.relu(h12, True)
    # RepeatStart_5
    for i in range(1):

        # Convolution_27
        h13 = PF.convolution(h12, 512, (3,3), (1,1), with_bias=False, name='Convolution_27[' + str(i) + ']')
        # BatchNormalization_5
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_5[' + str(i) + ']')
        # PReLU
        h13 = PF.prelu(h13, 1, False, name='PReLU[' + str(i) + ']')
        # DepthwiseConvolution_2
        h13 = PF.depthwise_convolution(h13, (5,5), (2,2), name='DepthwiseConvolution_2[' + str(i) + ']')
        # BatchNormalization
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization[' + str(i) + ']')
        # ReLU
        h13 = F.relu(h13, True)
        # DepthwiseConvolution
        h13 = PF.depthwise_convolution(h13, (5,5), (2,2), name='DepthwiseConvolution[' + str(i) + ']')
        # BatchNormalization_27
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_27[' + str(i) + ']')
        # ReLU_23
        h13 = F.relu(h13, True)
        # Convolution_28
        h13 = PF.convolution(h13, 512, (3,3), (1,1), with_bias=False, name='Convolution_28[' + str(i) + ']')
        # BatchNormalization_28
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_28[' + str(i) + ']')

        # Add2_7 -> 512,2,2
        h14 = F.add2(h12, h13, False)
        # ReLU_25
        h14 = F.relu(h14, True)
        # RepeatEnd_5
        h12 = h14
    # Affine_2 -> 2
    h14 = PF.affine(h14, (2,), name='Affine_2')
    # SquaredError_2
    #h14 = F.squared_error(h14, y)
    return h14

def network4(x, y, test=False):
    # InputX:x -> 1,480,245
    # MulScalar_2
    h = F.mul_scalar(x, 0.003921568627451)
    # ImageAugmentation
    if not test:
        h = F.image_augmentation(h, (1,480,245), (0,0), 1.0, 1.0, 0.1, 1.0, 0.0, False, False, 0.0, False, 1.0, 0.5, False, 0.0)
    # AveragePooling -> 1,60,49
    h = F.average_pooling(h, (8,5), (8,5), False)
    # Convolution_2 -> 64,30,25
    h = PF.convolution(h, 64, (7,7), (3,3), (2,2), with_bias=False, name='Convolution_2')
    # BatchNormalization_2
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # ReLU_2
    h = F.relu(h, True)
    # MaxPooling_2 -> 64,11,12
    h = F.max_pooling(h, (11,4), (2,2), True, (1,1))

    # Convolution_3
    h1 = PF.convolution(h, 64, (3,3), (1,1), with_bias=False, name='Convolution_3')
    # BatchNormalization_3
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # ReLU_4
    h1 = F.relu(h1, True)
    # Convolution_5
    h1 = PF.convolution(h1, 64, (3,3), (1,1), with_bias=False, name='Convolution_5')
    # BatchNormalization_4
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')

    # Add2 -> 64,11,12
    h2 = F.add2(h, h1, False)
    # ReLU_5
    h2 = F.relu(h2, True)
    # RepeatStart_2
    for i in range(2):

        # Convolution_6 -> 93,11,12
        h3 = PF.convolution(h2, 93, (3,3), (1,1), with_bias=False, name='Convolution_6[' + str(i) + ']')
        # BatchNormalization_6
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6[' + str(i) + ']')
        # ReLU_6
        h3 = F.relu(h3, True)
        # Convolution_7 -> 64,11,12
        h3 = PF.convolution(h3, 64, (3,3), (1,1), with_bias=False, name='Convolution_7[' + str(i) + ']')
        # BatchNormalization_7
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_7[' + str(i) + ']')

        # Add2_2 -> 64,11,12
        h4 = F.add2(h2, h3, False)
        # ReLU_7
        h4 = F.relu(h4, True)
        # RepeatEnd_2
        h2 = h4

    # Convolution_10 -> 126,6,6
    h5 = PF.convolution(h4, 126, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_10')

    # Convolution_9 -> 128,6,6
    h6 = PF.convolution(h4, 128, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_9')
    # BatchNormalization_10
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_10')
    # BatchNormalization_9
    h6 = PF.batch_normalization(h6, (1,), 0.9, 0.0001, not test, name='BatchNormalization_9')
    # ELU
    h5 = F.elu(h5, 1.0)
    # Convolution_11 -> 128,6,6
    h5 = PF.convolution(h5, 128, (5,3), (2,1), with_bias=False, name='Convolution_11')
    # BatchNormalization_11
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_11')

    # Add2_3 -> 128,6,6
    h6 = F.add2(h6, h5, True)
    # ReLU_10
    h6 = F.relu(h6, True)
    # RepeatStart_3
    for i in range(2):

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

        # Add2_4 -> 128,6,6
        h8 = F.add2(h6, h7, False)
        # ReLU_13
        h8 = F.relu(h8, True)
        # RepeatEnd_3
        h6 = h8

    # Convolution_17 -> 228,3,3
    h9 = PF.convolution(h8, 228, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_17')

    # Convolution_16 -> 256,3,3
    h10 = PF.convolution(h8, 256, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_16')
    # BatchNormalization_17
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_17')
    # BatchNormalization_16
    h10 = PF.batch_normalization(h10, (1,), 0.9, 0.0001, not test, name='BatchNormalization_16')
    # ReLU_14
    h9 = F.relu(h9, True)
    # Convolution_18 -> 256,3,3
    h9 = PF.convolution(h9, 256, (3,3), (1,1), with_bias=False, name='Convolution_18')
    # BatchNormalization_18
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_18')

    # Add2_5 -> 256,3,3
    h10 = F.add2(h10, h9, True)
    # ReLU_16
    h10 = F.relu(h10, True)
    # RepeatStart_4
    for i in range(1):
        # ReLU_19
        h10 = F.relu(h10, False)
        # RepeatEnd_4
    # Convolution_23 -> 512,2,2
    h10 = PF.convolution(h10, 512, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_23')
    # BatchNormalization_23
    h10 = PF.batch_normalization(h10, (1,), 0.9, 0.0001, not test, name='BatchNormalization_23')
    # ReLU_22
    h10 = F.relu(h10, True)
    # RepeatStart_5
    for i in range(1):

        # Convolution_27
        h11 = PF.convolution(h10, 512, (3,3), (1,1), with_bias=False, name='Convolution_27[' + str(i) + ']')
        # BatchNormalization_27
        h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_27[' + str(i) + ']')
        # ReLU_23
        h11 = F.relu(h11, True)
        # Convolution_28
        h11 = PF.convolution(h11, 512, (3,3), (1,1), with_bias=False, name='Convolution_28[' + str(i) + ']')
        # BatchNormalization_28
        h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_28[' + str(i) + ']')

        # Add2_7 -> 512,2,2
        h12 = F.add2(h10, h11, False)
        # ReLU_25
        h12 = F.relu(h12, True)
        # RepeatEnd_5
        h10 = h12
    # AveragePooling_2 -> 512,1,1
    h12 = F.average_pooling(h12, (2,2), (2,2))
    # Affine_2 -> 2
    h12 = PF.affine(h12, (2,), name='Affine_2')
    # SquaredError_2
    #h12 = F.squared_error(h12, y)
    return h12

def network5(x, y, test=False):
    # InputX:x -> 1,480,245
    # MulScalar_2
    h = F.mul_scalar(x, 0.003921568627451)
    # ImageAugmentation
    if not test:
        h = F.image_augmentation(h, (1,480,245), (0,0), 1.0, 1.0, 0.1, 1.0, 0.0, False, False, 0.0, False, 1.0, 0.5, False, 0.0)
    # AveragePooling -> 1,37,35
    h = F.average_pooling(h, (13,7), (13,7), False)
    # Convolution_2 -> 64,19,18
    h = PF.convolution(h, 64, (7,7), (3,3), (2,2), with_bias=False, name='Convolution_2')
    # BatchNormalization_2
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # ReLU_2
    h = F.relu(h, True)
    # MaxPooling_2 -> 64,6,9
    h = F.max_pooling(h, (11,4), (2,2), True, (1,1))

    # Convolution_3
    h1 = PF.convolution(h, 64, (3,3), (1,1), with_bias=False, name='Convolution_3')
    # BatchNormalization_3
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # ReLU_4
    h1 = F.relu(h1, True)
    # Convolution_5
    h1 = PF.convolution(h1, 64, (3,3), (1,1), with_bias=False, name='Convolution_5')
    # BatchNormalization_4
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')

    # Add2 -> 64,6,9
    h2 = F.add2(h, h1, False)
    # ReLU_5
    h2 = F.relu(h2, True)
    # RepeatStart_2
    for i in range(2):

        # Convolution_6 -> 93,6,9
        h3 = PF.convolution(h2, 93, (3,3), (1,1), with_bias=False, name='Convolution_6[' + str(i) + ']')
        # BatchNormalization_6
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6[' + str(i) + ']')
        # ReLU_6
        h3 = F.relu(h3, True)
        # Convolution_7 -> 64,6,9
        h3 = PF.convolution(h3, 64, (3,3), (1,1), with_bias=False, name='Convolution_7[' + str(i) + ']')
        # BatchNormalization_7
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_7[' + str(i) + ']')

        # Add2_2 -> 64,6,9
        h4 = F.add2(h2, h3, False)
        # ReLU_7
        h4 = F.relu(h4, True)
        # RepeatEnd_2
        h2 = h4

    # Convolution_10 -> 128,3,5
    h5 = PF.convolution(h4, 128, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_10')

    # Convolution_9 -> 128,3,5
    h6 = PF.convolution(h4, 128, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_9')
    # BatchNormalization_10
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_10')
    # BatchNormalization_9
    h6 = PF.batch_normalization(h6, (1,), 0.9, 0.0001, not test, name='BatchNormalization_9')
    # ReLU_8
    h5 = F.relu(h5, True)
    # Convolution_11
    h5 = PF.convolution(h5, 128, (5,3), (2,1), with_bias=False, name='Convolution_11')
    # BatchNormalization_11
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_11')

    # Add2_3 -> 128,3,5
    h6 = F.add2(h6, h5, True)
    # ReLU_10
    h6 = F.relu(h6, True)
    # RepeatStart_3
    for i in range(2):

        # Convolution_13 -> 65,3,5
        h7 = PF.convolution(h6, 65, (3,3), (1,1), with_bias=False, name='Convolution_13[' + str(i) + ']')
        # BatchNormalization_13
        h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_13[' + str(i) + ']')
        # ReLU_11
        h7 = F.relu(h7, True)
        # Convolution_14 -> 128,3,5
        h7 = PF.convolution(h7, 128, (3,3), (1,1), with_bias=False, name='Convolution_14[' + str(i) + ']')
        # BatchNormalization_14
        h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_14[' + str(i) + ']')

        # Add2_4 -> 128,3,5
        h8 = F.add2(h6, h7, False)
        # ReLU_13
        h8 = F.relu(h8, True)
        # RepeatEnd_3
        h6 = h8

    # Convolution_17 -> 223,2,3
    h9 = PF.convolution(h8, 223, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_17')

    # Dropout
    if not test:
        h10 = F.dropout(h8)
    else:
        h10 = h8
    # BatchNormalization_17
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_17')
    # Convolution_16 -> 256,2,3
    h10 = PF.convolution(h10, 256, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_16')
    # ReLU_14
    h9 = F.relu(h9, True)
    # BatchNormalization_16
    h10 = PF.batch_normalization(h10, (1,), 0.9, 0.0001, not test, name='BatchNormalization_16')
    # Convolution_18 -> 256,2,3
    h9 = PF.convolution(h9, 256, (3,3), (1,1), with_bias=False, name='Convolution_18')
    # BatchNormalization_18
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_18')

    # Add2_5 -> 256,2,3
    h10 = F.add2(h10, h9, True)
    # ReLU_16
    h10 = F.relu(h10, True)
    # RepeatStart_4
    for i in range(3):
        # ReLU_19
        h10 = F.relu(h10, False)
        # RepeatEnd_4
    # Convolution_23 -> 512,1,2
    h10 = PF.convolution(h10, 512, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_23')
    # BatchNormalization_23
    h10 = PF.batch_normalization(h10, (1,), 0.9, 0.0001, not test, name='BatchNormalization_23')
    # ReLU_22
    h10 = F.relu(h10, True)
    # RepeatStart_5
    for i in range(1):

        # Convolution_27
        h11 = PF.convolution(h10, 512, (3,3), (1,1), with_bias=False, name='Convolution_27[' + str(i) + ']')
        # BatchNormalization_27
        h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_27[' + str(i) + ']')
        # PReLU
        h11 = PF.prelu(h11, 1, False, name='PReLU[' + str(i) + ']')
        # Convolution_28
        h11 = PF.convolution(h11, 512, (3,3), (1,1), with_bias=False, name='Convolution_28[' + str(i) + ']')
        # BatchNormalization_28
        h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_28[' + str(i) + ']')

        # Add2_7 -> 512,1,2
        h12 = F.add2(h10, h11, False)
        # ReLU_25
        h12 = F.relu(h12, True)
        # RepeatEnd_5
        h10 = h12
    # Affine_2 -> 2
    h12 = PF.affine(h12, (2,), name='Affine_2')
    # SquaredError_2
    #h12 = F.squared_error(h12, y)
    return h12

def network6(x, y, test=False):
    # InputX:x -> 1,480,245
    # MulScalar_2
    h = F.mul_scalar(x, 0.003921568627451)
    # ImageAugmentation
    if not test:
        h = F.image_augmentation(h, (1,480,245), (0,0), 0.995, 1.02, 0.1, 1.0, 0.0, False, False, 0.03, False, 1.0, 0.5, False, 0.0)
    # AveragePooling -> 1,40,35
    h = F.average_pooling(h, (12,7), (12,7), False)
    # Convolution_2 -> 64,20,18
    h = PF.convolution(h, 64, (7,7), (3,3), (2,2), with_bias=False, name='Convolution_2')
    # BatchNormalization_2
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # ReLU_2
    h = F.relu(h, True)
    # AveragePooling_2 -> 64,6,9
    h = F.average_pooling(h, (11,4), (2,2), True, (1,1))
    # ReLU_5
    h = F.relu(h, True)
    # RepeatStart_2
    for i in range(2):

        # Convolution_6 -> 43,6,9
        h1 = PF.convolution(h, 43, (3,3), (1,1), with_bias=False, name='Convolution_6[' + str(i) + ']')
        # BatchNormalization_6
        h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6[' + str(i) + ']')
        # ReLU_6
        h1 = F.relu(h1, True)
        # Convolution_7 -> 64,6,9
        h1 = PF.convolution(h1, 64, (3,3), (1,1), with_bias=False, name='Convolution_7[' + str(i) + ']')
        # BatchNormalization_7
        h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_7[' + str(i) + ']')

        # Add2_2 -> 64,6,9
        h2 = F.add2(h, h1, False)
        # ReLU_7
        h2 = F.relu(h2, True)
        # RepeatEnd_2
        h = h2

    # Convolution_10 -> 128,3,5
    h3 = PF.convolution(h2, 128, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_10')

    # Convolution_9 -> 128,3,5
    h4 = PF.convolution(h2, 128, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_9')
    # BatchNormalization_10
    h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_10')
    # BatchNormalization_9
    h4 = PF.batch_normalization(h4, (1,), 0.9, 0.0001, not test, name='BatchNormalization_9')
    # ReLU_8
    h3 = F.relu(h3, True)
    # Convolution_11
    h3 = PF.convolution(h3, 128, (5,3), (2,1), with_bias=False, name='Convolution_11')
    # BatchNormalization_11
    h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_11')

    # Add2_3 -> 128,3,5
    h4 = F.add2(h4, h3, True)
    # ReLU_10
    h4 = F.relu(h4, True)
    # RepeatStart_3
    for i in range(2):

        # Convolution_13
        h5 = PF.convolution(h4, 128, (3,3), (1,1), with_bias=False, name='Convolution_13[' + str(i) + ']')
        # BatchNormalization_13
        h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_13[' + str(i) + ']')
        # ReLU_11
        h5 = F.relu(h5, True)
        # Convolution_14
        h5 = PF.convolution(h5, 128, (3,3), (1,1), with_bias=False, name='Convolution_14[' + str(i) + ']')
        # BatchNormalization_14
        h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_14[' + str(i) + ']')

        # Add2_4 -> 128,3,5
        h6 = F.add2(h4, h5, False)
        # ReLU_13
        h6 = F.relu(h6, True)
        # RepeatEnd_3
        h4 = h6

    # Convolution_17 -> 256,2,3
    h7 = PF.convolution(h6, 256, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_17')

    # Convolution_16 -> 256,2,3
    h8 = PF.convolution(h6, 256, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_16')
    # BatchNormalization_17
    h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_17')
    # BatchNormalization_16
    h8 = PF.batch_normalization(h8, (1,), 0.9, 0.0001, not test, name='BatchNormalization_16')
    # ReLU_14
    h7 = F.relu(h7, True)
    # Convolution_18
    h7 = PF.convolution(h7, 256, (3,3), (1,1), with_bias=False, name='Convolution_18')
    # BatchNormalization_18
    h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_18')

    # Add2_5 -> 256,2,3
    h8 = F.add2(h8, h7, True)
    # ReLU_16
    h8 = F.relu(h8, True)
    # RepeatStart_4
    for i in range(1):
        # ReLU_19
        h8 = F.relu(h8, False)
        # RepeatEnd_4

    # Convolution_24 -> 512,1,2
    h9 = PF.convolution(h8, 512, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_24')

    # Convolution_23 -> 512,1,2
    h10 = PF.convolution(h8, 512, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_23')
    # BatchNormalization_24
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_24')
    # BatchNormalization_23
    h10 = PF.batch_normalization(h10, (1,), 0.9, 0.0001, not test, name='BatchNormalization_23')
    # ReLU_20
    h9 = F.relu(h9, True)
    # Convolution_25
    h9 = PF.convolution(h9, 512, (3,3), (1,1), with_bias=False, name='Convolution_25')
    # BatchNormalization_25
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_25')

    # Add2_8 -> 512,1,2
    h10 = F.add2(h10, h9, True)
    # ReLU_22
    h10 = F.relu(h10, True)
    # RepeatStart_5
    for i in range(1):

        # Convolution_27
        h11 = PF.convolution(h10, 512, (3,3), (1,1), with_bias=False, name='Convolution_27[' + str(i) + ']')
        # BatchNormalization_5
        h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_5[' + str(i) + ']')
        # PReLU
        h11 = PF.prelu(h11, 1, False, name='PReLU[' + str(i) + ']')
        # DepthwiseConvolution_2
        h11 = PF.depthwise_convolution(h11, (5,5), (2,2), name='DepthwiseConvolution_2[' + str(i) + ']')
        # BatchNormalization
        h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization[' + str(i) + ']')
        # ReLU
        h11 = F.relu(h11, True)
        # DepthwiseConvolution
        h11 = PF.depthwise_convolution(h11, (5,5), (2,2), name='DepthwiseConvolution[' + str(i) + ']')
        # BatchNormalization_27
        h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_27[' + str(i) + ']')
        # ReLU_23
        h11 = F.relu(h11, True)
        # Convolution_28
        h11 = PF.convolution(h11, 512, (3,3), (1,1), with_bias=False, name='Convolution_28[' + str(i) + ']')
        # BatchNormalization_28
        h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_28[' + str(i) + ']')

        # Add2_7 -> 512,1,2
        h12 = F.add2(h10, h11, False)
        # ReLU_25
        h12 = F.relu(h12, True)
        # RepeatEnd_5
        h10 = h12
    # Affine_2 -> 2
    h12 = PF.affine(h12, (2,), name='Affine_2')
    # SquaredError_2
    #h12 = F.squared_error(h12, y)
    return h12

def network7(x, y, test=False):
    # InputX:x -> 1,480,245
    # MulScalar_2
    h = F.mul_scalar(x, 0.003921568627451)
    # ImageAugmentation
    if not test:
        h = F.image_augmentation(h, (1,480,245), (0,0), 1.0, 1.0, 0.1, 1.0, 0.0, False, False, 0.0, False, 1.0, 0.5, False, 0.0)
    # AveragePooling -> 1,60,62
    h = F.average_pooling(h, (8,4), (8,4), False)
    # Convolution_2 -> 64,30,31
    h = PF.convolution(h, 64, (7,7), (3,3), (2,2), with_bias=False, name='Convolution_2')
    # BatchNormalization_2
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # ReLU_2
    h = F.relu(h, True)
    # AveragePooling_2 -> 64,8,15
    h = F.average_pooling(h, (17,4), (2,2), True, (1,1))

    # Convolution_3 -> 67,8,15
    h1 = PF.convolution(h, 67, (1,3), (0,1), with_bias=False, name='Convolution_3')
    # BatchNormalization_3
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # ReLU_4
    h1 = F.relu(h1, True)
    # Convolution_5 -> 64,8,15
    h1 = PF.convolution(h1, 64, (3,3), (1,1), with_bias=False, name='Convolution_5')
    # BatchNormalization_4
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')

    # Add2 -> 64,8,15
    h2 = F.add2(h, h1, False)
    # ReLU_5
    h2 = F.relu(h2, True)
    # RepeatStart_2
    for i in range(2):

        # Convolution_6 -> 43,8,15
        h3 = PF.convolution(h2, 43, (1,1), (0,0), with_bias=False, name='Convolution_6[' + str(i) + ']')
        # BatchNormalization_6
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6[' + str(i) + ']')
        # ReLU_6
        h3 = F.relu(h3, True)
        # Convolution_7 -> 64,8,15
        h3 = PF.convolution(h3, 64, (3,1), (1,0), with_bias=False, name='Convolution_7[' + str(i) + ']')
        # BatchNormalization_7
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_7[' + str(i) + ']')

        # Add2_2 -> 64,8,15
        h4 = F.add2(h2, h3, False)
        # ReLU_7
        h4 = F.relu(h4, True)
        # RepeatEnd_2
        h2 = h4

    # Convolution_10 -> 128,4,8
    h5 = PF.convolution(h4, 128, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_10')

    # Convolution_9 -> 128,4,8
    h6 = PF.convolution(h4, 128, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_9')
    # BatchNormalization_10
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_10')
    # BatchNormalization_9
    h6 = PF.batch_normalization(h6, (1,), 0.9, 0.0001, not test, name='BatchNormalization_9')
    # ReLU_8
    h5 = F.relu(h5, True)
    # Convolution_11
    h5 = PF.convolution(h5, 128, (5,3), (2,1), with_bias=False, name='Convolution_11')
    # BatchNormalization_11
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_11')

    # Add2_3 -> 128,4,8
    h6 = F.add2(h6, h5, True)
    # ReLU_10
    h6 = F.relu(h6, True)
    # RepeatStart_3
    for i in range(1):

        # Convolution_13
        h7 = PF.convolution(h6, 128, (3,3), (1,1), with_bias=False, name='Convolution_13[' + str(i) + ']')
        # BatchNormalization_13
        h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_13[' + str(i) + ']')
        # ReLU_11
        h7 = F.relu(h7, True)
        # Convolution_14
        h7 = PF.convolution(h7, 128, (1,3), (0,1), with_bias=False, name='Convolution_14[' + str(i) + ']')
        # BatchNormalization_14
        h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_14[' + str(i) + ']')

        # Add2_4 -> 128,4,8
        h8 = F.add2(h6, h7, False)
        # ReLU_13
        h8 = F.relu(h8, True)
        # RepeatEnd_3
        h6 = h8

    # Convolution_17 -> 256,2,4
    h9 = PF.convolution(h8, 256, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_17')

    # Convolution_16 -> 256,2,4
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

    # Add2_5 -> 256,2,4
    h10 = F.add2(h10, h9, True)
    # ReLU_16
    h10 = F.relu(h10, True)
    # RepeatStart_4
    for i in range(1):
        # ReLU_19
        h10 = F.relu(h10, False)
        # RepeatEnd_4

    # Convolution_24 -> 512,1,2
    h11 = PF.convolution(h10, 512, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_24')

    # Convolution_23 -> 512,1,2
    h12 = PF.convolution(h10, 512, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_23')
    # BatchNormalization_24
    h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_24')
    # BatchNormalization_23
    h12 = PF.batch_normalization(h12, (1,), 0.9, 0.0001, not test, name='BatchNormalization_23')
    # ReLU_20
    h11 = F.relu(h11, True)
    # Convolution_25
    h11 = PF.convolution(h11, 512, (1,3), (0,1), with_bias=False, name='Convolution_25')
    # BatchNormalization_25
    h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_25')

    # Add2_8 -> 512,1,2
    h12 = F.add2(h12, h11, True)
    # ReLU_22
    h12 = F.relu(h12, True)
    # RepeatStart_5
    for i in range(1):

        # Convolution_27 -> 133,1,2
        h13 = PF.convolution(h12, 133, (3,3), (1,1), with_bias=False, name='Convolution_27[' + str(i) + ']')
        # BatchNormalization_5
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_5[' + str(i) + ']')
        # PReLU
        h13 = PF.prelu(h13, 1, False, name='PReLU[' + str(i) + ']')
        # DepthwiseConvolution_2
        h13 = PF.depthwise_convolution(h13, (5,5), (2,2), name='DepthwiseConvolution_2[' + str(i) + ']')
        # BatchNormalization
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization[' + str(i) + ']')
        # ReLU
        h13 = F.relu(h13, True)
        # DepthwiseConvolution
        h13 = PF.depthwise_convolution(h13, (5,5), (2,2), name='DepthwiseConvolution[' + str(i) + ']')
        # BatchNormalization_27
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_27[' + str(i) + ']')
        # ReLU_23
        h13 = F.relu(h13, True)
        # Convolution_28 -> 512,1,2
        h13 = PF.convolution(h13, 512, (3,3), (1,1), with_bias=False, name='Convolution_28[' + str(i) + ']')
        # BatchNormalization_28
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_28[' + str(i) + ']')

        # Add2_7 -> 512,1,2
        h14 = F.add2(h12, h13, False)
        # ReLU_25
        h14 = F.relu(h14, True)
        # RepeatEnd_5
        h12 = h14
    # Affine_2 -> 2
    h14 = PF.affine(h14, (2,), name='Affine_2')
    # SquaredError_2
    #h14 = F.squared_error(h14, y)
    return h14

def network8(x, y, test=False):
    # InputX:x -> 1,480,245
    # MulScalar_2
    h = F.mul_scalar(x, 0.003921568627451)
    # ImageAugmentation
    if not test:
        h = F.image_augmentation(h, (1,480,245), (0,0), 1.0, 1.0, 0.1, 1.0, 0.0, False, False, 0.0, False, 1.0, 0.5, False, 0.0)
    # AveragePooling -> 1,60,62
    h = F.average_pooling(h, (8,4), (8,4), False)
    # Convolution_2 -> 64,30,31
    h = PF.convolution(h, 64, (7,7), (3,3), (2,2), with_bias=False, name='Convolution_2')
    # BatchNormalization_2
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # ReLU_2
    h = F.relu(h, True)
    # AveragePooling_2 -> 64,8,15
    h = F.average_pooling(h, (17,4), (2,2), True, (1,1))

    # Convolution_3 -> 68,8,15
    h1 = PF.convolution(h, 68, (3,3), (1,1), with_bias=False, name='Convolution_3')
    # BatchNormalization_3
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # ReLU_4
    h1 = F.relu(h1, True)
    # Convolution_5 -> 64,8,15
    h1 = PF.convolution(h1, 64, (3,3), (1,1), with_bias=False, name='Convolution_5')
    # BatchNormalization_4
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')

    # Add2 -> 64,8,15
    h2 = F.add2(h, h1, False)
    # ReLU_5
    h2 = F.relu(h2, True)
    # RepeatStart_2
    for i in range(2):

        # Convolution_6 -> 43,8,15
        h3 = PF.convolution(h2, 43, (1,1), (0,0), with_bias=False, name='Convolution_6[' + str(i) + ']')
        # BatchNormalization_6
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6[' + str(i) + ']')
        # ReLU_6
        h3 = F.relu(h3, True)
        # Convolution_7 -> 64,8,15
        h3 = PF.convolution(h3, 64, (3,1), (1,0), with_bias=False, name='Convolution_7[' + str(i) + ']')
        # BatchNormalization_7
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_7[' + str(i) + ']')

        # Add2_2 -> 64,8,15
        h4 = F.add2(h2, h3, False)
        # ReLU_7
        h4 = F.relu(h4, True)
        # RepeatEnd_2
        h2 = h4
    # Convolution_9 -> 128,4,8
    h4 = PF.convolution(h4, 128, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_9')
    # BatchNormalization_9
    h4 = PF.batch_normalization(h4, (1,), 0.9, 0.0001, not test, name='BatchNormalization_9')
    # ReLU_10
    h4 = F.relu(h4, True)
    # RepeatStart_3
    for i in range(1):

        # Convolution_13
        h5 = PF.convolution(h4, 128, (1,1), (0,0), with_bias=False, name='Convolution_13[' + str(i) + ']')
        # BatchNormalization_13
        h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_13[' + str(i) + ']')
        # ReLU_11
        h5 = F.relu(h5, True)
        # Convolution_14
        h5 = PF.convolution(h5, 128, (1,3), (0,1), with_bias=False, name='Convolution_14[' + str(i) + ']')
        # BatchNormalization_14
        h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_14[' + str(i) + ']')

        # Add2_4 -> 128,4,8
        h6 = F.add2(h4, h5, False)
        # ReLU_13
        h6 = F.relu(h6, True)
        # RepeatEnd_3
        h4 = h6

    # Convolution_17 -> 256,2,4
    h7 = PF.convolution(h6, 256, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_17')

    # Convolution_16 -> 256,2,4
    h8 = PF.convolution(h6, 256, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_16')
    # BatchNormalization_17
    h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_17')
    # BatchNormalization_16
    h8 = PF.batch_normalization(h8, (1,), 0.9, 0.0001, not test, name='BatchNormalization_16')
    # ReLU_14
    h7 = F.relu(h7, True)
    # Convolution_18
    h7 = PF.convolution(h7, 256, (3,3), (1,1), with_bias=False, name='Convolution_18')
    # BatchNormalization_18
    h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_18')

    # Add2_5 -> 256,2,4
    h8 = F.add2(h8, h7, True)
    # ReLU_16
    h8 = F.relu(h8, True)
    # RepeatStart_4
    for i in range(1):
        # ReLU_19
        h8 = F.relu(h8, False)
        # RepeatEnd_4

    # Convolution_24 -> 508,1,2
    h9 = PF.convolution(h8, 508, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_24')

    # Convolution_23 -> 512,1,2
    h10 = PF.convolution(h8, 512, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_23')
    # BatchNormalization_24
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_24')
    # BatchNormalization_23
    h10 = PF.batch_normalization(h10, (1,), 0.9, 0.0001, not test, name='BatchNormalization_23')
    # ReLU_20
    h9 = F.relu(h9, True)
    # Convolution_25 -> 512,1,2
    h9 = PF.convolution(h9, 512, (1,3), (0,1), with_bias=False, name='Convolution_25')
    # BatchNormalization_25
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_25')

    # Add2_8 -> 512,1,2
    h10 = F.add2(h10, h9, True)
    # ReLU_22
    h10 = F.relu(h10, True)
    # RepeatStart_5
    for i in range(1):

        # Convolution_27 -> 265,1,2
        h11 = PF.convolution(h10, 265, (3,1), (1,0), with_bias=False, name='Convolution_27[' + str(i) + ']')
        # BatchNormalization_5
        h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_5[' + str(i) + ']')
        # PReLU
        h11 = PF.prelu(h11, 1, False, name='PReLU[' + str(i) + ']')
        # DepthwiseConvolution_2
        h11 = PF.depthwise_convolution(h11, (3,3), (1,1), name='DepthwiseConvolution_2[' + str(i) + ']')
        # BatchNormalization
        h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization[' + str(i) + ']')
        # ReLU
        h11 = F.relu(h11, True)
        # Convolution_28 -> 512,1,2
        h11 = PF.convolution(h11, 512, (3,3), (1,1), with_bias=False, name='Convolution_28[' + str(i) + ']')
        # BatchNormalization_28
        h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_28[' + str(i) + ']')

        # Add2_7 -> 512,1,2
        h12 = F.add2(h10, h11, False)
        # ReLU_25
        h12 = F.relu(h12, True)
        # RepeatEnd_5
        h10 = h12
    # Affine_2 -> 2
    h12 = PF.affine(h12, (2,), name='Affine_2')
    # SquaredError_2
    #h12 = F.squared_error(h12, y)
    return h12

def network9(x, y, test=False):
    # InputX:x -> 1,480,245
    # MulScalar_2
    h = F.mul_scalar(x, 0.003921568627451)
    # ImageAugmentation
    if not test:
        h = F.image_augmentation(h, (1,480,245), (0,0), 1.0, 1.0, 0.1, 1.0, 0.0, False, False, 0.0, False, 1.0, 0.5, False, 0.0)
    # AveragePooling -> 1,54,49
    h = F.average_pooling(h, (9,5), (9,5), False)
    # Convolution_2 -> 64,27,25
    h = PF.convolution(h, 64, (7,7), (3,3), (2,2), with_bias=False, name='Convolution_2')
    # BatchNormalization_2
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # ReLU_2
    h = F.relu(h, True)
    # AveragePooling_2 -> 64,7,12
    h = F.average_pooling(h, (17,4), (2,2), True, (1,1))

    # Convolution_3 -> 67,7,12
    h1 = PF.convolution(h, 67, (1,3), (0,1), with_bias=False, name='Convolution_3')
    # BatchNormalization_3
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # ReLU_4
    h1 = F.relu(h1, True)
    # Convolution_5 -> 64,7,12
    h1 = PF.convolution(h1, 64, (3,3), (1,1), with_bias=False, name='Convolution_5')
    # BatchNormalization_4
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')

    # Add2 -> 64,7,12
    h2 = F.add2(h, h1, False)
    # ReLU_5
    h2 = F.relu(h2, True)
    # RepeatStart_2
    for i in range(2):

        # Convolution_6 -> 43,7,12
        h3 = PF.convolution(h2, 43, (1,1), (0,0), with_bias=False, name='Convolution_6[' + str(i) + ']')
        # BatchNormalization_6
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6[' + str(i) + ']')
        # ReLU_6
        h3 = F.relu(h3, True)
        # Convolution_7 -> 64,7,12
        h3 = PF.convolution(h3, 64, (1,1), (0,0), with_bias=False, name='Convolution_7[' + str(i) + ']')
        # BatchNormalization_7
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_7[' + str(i) + ']')

        # Add2_2 -> 64,7,12
        h4 = F.add2(h2, h3, False)
        # ReLU_7
        h4 = F.relu(h4, True)
        # RepeatEnd_2
        h2 = h4

    # Convolution_10 -> 128,4,6
    h5 = PF.convolution(h4, 128, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_10')

    # Convolution_9 -> 128,4,6
    h6 = PF.convolution(h4, 128, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_9')
    # BatchNormalization_10
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_10')
    # BatchNormalization_8
    h6 = PF.batch_normalization(h6, (1,), 0.9, 0.0001, not test, name='BatchNormalization_8')
    # ReLU_8
    h5 = F.relu(h5, True)
    # SELU
    h6 = F.selu(h6)
    # Convolution_11
    h5 = PF.convolution(h5, 128, (5,3), (2,1), with_bias=False, name='Convolution_11')
    # DepthwiseConvolution_3
    h6 = PF.depthwise_convolution(h6, (5,5), (2,2), name='DepthwiseConvolution_3')
    # BatchNormalization_11
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_11')
    # BatchNormalization_9
    h6 = PF.batch_normalization(h6, (1,), 0.9, 0.0001, not test, name='BatchNormalization_9')

    # Add2_3 -> 128,4,6
    h6 = F.add2(h6, h5, True)
    # ReLU_10
    h6 = F.relu(h6, True)
    # RepeatStart_3
    for i in range(1):

        # Convolution_13
        h7 = PF.convolution(h6, 128, (3,3), (1,1), with_bias=False, name='Convolution_13[' + str(i) + ']')
        # BatchNormalization_13
        h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_13[' + str(i) + ']')
        # ReLU_11
        h7 = F.relu(h7, True)
        # Convolution_14
        h7 = PF.convolution(h7, 128, (1,3), (0,1), with_bias=False, name='Convolution_14[' + str(i) + ']')
        # BatchNormalization_14
        h7 = PF.batch_normalization(h7, (1,), 0.9, 0.0001, not test, name='BatchNormalization_14[' + str(i) + ']')

        # Add2_4 -> 128,4,6
        h8 = F.add2(h6, h7, False)
        # ReLU_13
        h8 = F.relu(h8, True)
        # RepeatEnd_3
        h6 = h8

    # Convolution_17 -> 256,2,3
    h9 = PF.convolution(h8, 256, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_17')

    # Convolution_16 -> 256,2,3
    h10 = PF.convolution(h8, 256, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_16')
    # BatchNormalization_17
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_17')
    # BatchNormalization_12
    h10 = PF.batch_normalization(h10, (1,), 0.9, 0.0001, not test, name='BatchNormalization_12')
    # ReLU_14
    h9 = F.relu(h9, True)
    # PReLU_2
    h10 = PF.prelu(h10, 1, False, name='PReLU_2')
    # Convolution_18
    h9 = PF.convolution(h9, 256, (3,3), (1,1), with_bias=False, name='Convolution_18')
    # DepthwiseConvolution_4
    h10 = PF.depthwise_convolution(h10, (5,5), (2,2), name='DepthwiseConvolution_4')
    # BatchNormalization_18
    h9 = PF.batch_normalization(h9, (1,), 0.9, 0.0001, not test, name='BatchNormalization_18')
    # BatchNormalization_16
    h10 = PF.batch_normalization(h10, (1,), 0.9, 0.0001, not test, name='BatchNormalization_16')

    # Add2_5 -> 256,2,3
    h10 = F.add2(h10, h9, True)
    # ReLU_16
    h10 = F.relu(h10, True)
    # RepeatStart_4
    for i in range(1):
        # ReLU_19
        h10 = F.relu(h10, False)
        # RepeatEnd_4

    # Convolution_24 -> 512,1,2
    h11 = PF.convolution(h10, 512, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_24')

    # Convolution_23 -> 512,1,2
    h12 = PF.convolution(h10, 512, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_23')
    # BatchNormalization_24
    h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_24')
    # BatchNormalization_23
    h12 = PF.batch_normalization(h12, (1,), 0.9, 0.0001, not test, name='BatchNormalization_23')
    # ReLU_20
    h11 = F.relu(h11, True)
    # Convolution_25
    h11 = PF.convolution(h11, 512, (1,3), (0,1), with_bias=False, name='Convolution_25')
    # BatchNormalization_25
    h11 = PF.batch_normalization(h11, (1,), 0.9, 0.0001, not test, name='BatchNormalization_25')

    # Add2_8 -> 512,1,2
    h12 = F.add2(h12, h11, True)
    # ReLU_22
    h12 = F.relu(h12, True)
    # RepeatStart_5
    for i in range(1):

        # Convolution_27 -> 133,1,2
        h13 = PF.convolution(h12, 133, (3,3), (1,1), with_bias=False, name='Convolution_27[' + str(i) + ']')
        # BatchNormalization_5
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_5[' + str(i) + ']')
        # PReLU
        h13 = PF.prelu(h13, 1, False, name='PReLU[' + str(i) + ']')
        # DepthwiseConvolution_2
        h13 = PF.depthwise_convolution(h13, (5,5), (2,2), name='DepthwiseConvolution_2[' + str(i) + ']')
        # BatchNormalization
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization[' + str(i) + ']')
        # ReLU
        h13 = F.relu(h13, True)
        # DepthwiseConvolution
        h13 = PF.depthwise_convolution(h13, (5,5), (2,2), name='DepthwiseConvolution[' + str(i) + ']')
        # BatchNormalization_27
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_27[' + str(i) + ']')
        # ReLU_23
        h13 = F.relu(h13, True)
        # Convolution_28 -> 512,1,2
        h13 = PF.convolution(h13, 512, (3,3), (1,1), with_bias=False, name='Convolution_28[' + str(i) + ']')
        # BatchNormalization_28
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_28[' + str(i) + ']')

        # Add2_7 -> 512,1,2
        h14 = F.add2(h12, h13, False)
        # ReLU_25
        h14 = F.relu(h14, True)
        # RepeatEnd_5
        h12 = h14
    # Affine_2 -> 2
    h14 = PF.affine(h14, (2,), name='Affine_2')
    # SquaredError_2
    #h14 = F.squared_error(h14, y)
    return h14

def network10(x, y, test=False):
    # InputX:x -> 1,480,245
    # MulScalar_2
    h = F.mul_scalar(x, 0.003921568627451)
    # ImageAugmentation
    if not test:
        h = F.image_augmentation(h, (1,480,245), (0,0), 1.0, 1.0, 0.1, 1.0, 0.0, False, False, 0.0, False, 1.0, 0.5, False, 0.0)
    # AveragePooling -> 1,60,49
    h = F.average_pooling(h, (8,5), (8,5), False)
    # Convolution_2 -> 64,30,25
    h = PF.convolution(h, 64, (7,7), (3,3), (2,2), with_bias=False, name='Convolution_2')
    # BatchNormalization_2
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization_2')
    # ReLU_2
    h = F.relu(h, True)
    # AveragePooling_2 -> 64,8,12
    h = F.average_pooling(h, (17,4), (2,2), True, (1,1))

    # Convolution_3 -> 68,8,12
    h1 = PF.convolution(h, 68, (3,3), (1,1), with_bias=False, name='Convolution_3')
    # BatchNormalization_3
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_3')
    # ReLU_4
    h1 = F.relu(h1, True)
    # Convolution_5 -> 64,8,12
    h1 = PF.convolution(h1, 64, (3,3), (1,1), with_bias=False, name='Convolution_5')
    # BatchNormalization_4
    h1 = PF.batch_normalization(h1, (1,), 0.9, 0.0001, not test, name='BatchNormalization_4')

    # Add2 -> 64,8,12
    h2 = F.add2(h, h1, False)
    # ReLU_5
    h2 = F.relu(h2, True)
    # RepeatStart_2
    for i in range(2):

        # Convolution_6 -> 43,8,12
        h3 = PF.convolution(h2, 43, (1,1), (0,0), with_bias=False, name='Convolution_6[' + str(i) + ']')
        # BatchNormalization_6
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_6[' + str(i) + ']')
        # ReLU_6
        h3 = F.relu(h3, True)
        # Convolution_7 -> 64,8,12
        h3 = PF.convolution(h3, 64, (3,3), (1,1), with_bias=False, name='Convolution_7[' + str(i) + ']')
        # BatchNormalization_7
        h3 = PF.batch_normalization(h3, (1,), 0.9, 0.0001, not test, name='BatchNormalization_7[' + str(i) + ']')

        # Add2_2 -> 64,8,12
        h4 = F.add2(h2, h3, False)
        # ReLU_7
        h4 = F.relu(h4, True)
        # RepeatEnd_2
        h2 = h4

    # Convolution_10 -> 128,4,6
    h5 = PF.convolution(h4, 128, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_10')

    # Convolution_9 -> 128,4,6
    h6 = PF.convolution(h4, 128, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_9')
    # BatchNormalization_10
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_10')
    # BatchNormalization_9
    h6 = PF.batch_normalization(h6, (1,), 0.9, 0.0001, not test, name='BatchNormalization_9')
    # ReLU_8
    h5 = F.relu(h5, True)
    # Convolution_11
    h5 = PF.convolution(h5, 128, (5,3), (2,1), with_bias=False, name='Convolution_11')
    # BatchNormalization_11
    h5 = PF.batch_normalization(h5, (1,), 0.9, 0.0001, not test, name='BatchNormalization_11')

    # Add2_3 -> 128,4,6
    h6 = F.add2(h6, h5, True)
    # ReLU_10
    h6 = F.relu(h6, True)
    # RepeatStart_3
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

        # Add2_4 -> 128,4,6
        h8 = F.add2(h6, h7, False)
        # ReLU_13
        h8 = F.relu(h8, True)
        # RepeatEnd_3
        h6 = h8

    # Convolution_17 -> 256,2,3
    h9 = PF.convolution(h8, 256, (3,3), (1,1), (2,2), with_bias=False, name='Convolution_17')

    # Convolution_16 -> 256,2,3
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

    # Add2_5 -> 256,2,3
    h10 = F.add2(h10, h9, True)
    # ReLU_16
    h10 = F.relu(h10, True)
    # RepeatStart_4
    for i in range(1):
        # ReLU_19
        h10 = F.relu(h10, False)
        # RepeatEnd_4

    # Convolution_24 -> 512,1,2
    h11 = PF.convolution(h10, 512, (1,1), (0,0), (2,2), with_bias=False, name='Convolution_24')

    # Convolution_23 -> 512,1,2
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

    # Add2_8 -> 512,1,2
    h12 = F.add2(h12, h11, True)
    # ReLU_22
    h12 = F.relu(h12, True)
    # RepeatStart_5
    for i in range(1):

        # Convolution_27
        h13 = PF.convolution(h12, 512, (3,3), (1,1), with_bias=False, name='Convolution_27[' + str(i) + ']')
        # BatchNormalization_5
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_5[' + str(i) + ']')
        # PReLU
        h13 = PF.prelu(h13, 1, False, name='PReLU[' + str(i) + ']')
        # DepthwiseConvolution_2
        h13 = PF.depthwise_convolution(h13, (5,5), (2,2), name='DepthwiseConvolution_2[' + str(i) + ']')
        # BatchNormalization
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization[' + str(i) + ']')
        # ReLU
        h13 = F.relu(h13, True)
        # DepthwiseConvolution
        h13 = PF.depthwise_convolution(h13, (5,5), (2,2), name='DepthwiseConvolution[' + str(i) + ']')
        # BatchNormalization_27
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_27[' + str(i) + ']')
        # ReLU_23
        h13 = F.relu(h13, True)
        # Convolution_28
        h13 = PF.convolution(h13, 512, (3,3), (1,1), with_bias=False, name='Convolution_28[' + str(i) + ']')
        # BatchNormalization_28
        h13 = PF.batch_normalization(h13, (1,), 0.9, 0.0001, not test, name='BatchNormalization_28[' + str(i) + ']')

        # Add2_7 -> 512,1,2
        h14 = F.add2(h12, h13, False)
        # ReLU_25
        h14 = F.relu(h14, True)
        # RepeatEnd_5
        h12 = h14
    # Affine_2 -> 2
    h14 = PF.affine(h14, (2,), name='Affine_2')
    # SquaredError_2
    #h14 = F.squared_error(h14, y)
    return h14

def network11(x, y, test=False):
    # InputX:x -> 1,480,123
    # MulScalar_2
    h = F.mul_scalar(x, val=0.003921568627451001)
    # ImageAugmentation
    if not test:
        h = F.image_augmentation(h, shape=(1,480,123), pad=(0,0), min_scale=1, max_scale=1, angle=0.1, aspect_ratio=1, distortion=0, brightness=0, contrast=1, noise=0)
    # AveragePooling -> 1,54,25
    h = F.average_pooling(h, kernel=(9,5), stride=(9,5), ignore_border=False)
    # Convolution_2 -> 64,27,13
    h = PF.convolution(h, outmaps=64, kernel=(7,7), pad=(3,3), stride=(2,2), with_bias=False, name='Convolution_2')
    # BatchNormalization_2
    h = PF.batch_normalization(h, batch_stat=not test, name='BatchNormalization_2')
    # ReLU_2
    h = F.relu(h, inplace=True)
    # AveragePooling_2 -> 64,1,7
    h = F.average_pooling(h, kernel=(29,2), stride=(2,2), pad=(1,1))

    # Convolution_5
    h1 = PF.convolution(h, outmaps=64, pad=(1,1), with_bias=False, name='Convolution_5')
    # BatchNormalization_4
    h1 = PF.batch_normalization(h1, batch_stat=not test, name='BatchNormalization_4')

    # Add2 -> 64,1,7
    h2 = F.add2(h, h1, inplace=False)
    # ReLU_5
    h2 = F.relu(h2, inplace=True)
    # RepeatStart_2
    for i in range(2):

        # Convolution_6 -> 43,1,7
        h3 = PF.convolution(h2, outmaps=43, kernel=(1,1), pad=(0,0), with_bias=False, name='Convolution_6[' + str(i) + ']')
        # BatchNormalization_6
        h3 = PF.batch_normalization(h3, batch_stat=not test, name='BatchNormalization_6[' + str(i) + ']')
        # ReLU_6
        h3 = F.relu(h3, inplace=True)
        # Dropout
        if not test:
            h3 = F.dropout(h3)
        # Convolution_7 -> 64,1,7
        h3 = PF.convolution(h3, outmaps=64, kernel=(1,1), pad=(0,0), with_bias=False, name='Convolution_7[' + str(i) + ']')
        # BatchNormalization_7
        h3 = PF.batch_normalization(h3, batch_stat=not test, name='BatchNormalization_7[' + str(i) + ']')

        # Add2_2 -> 64,1,7
        h4 = F.add2(h2, h3, inplace=False)
        # ReLU_7
        h4 = F.relu(h4, inplace=True)
        # RepeatEnd_2
        h2 = h4

    # Convolution_10 -> 128,1,4
    h5 = PF.convolution(h4, outmaps=128, pad=(1,1), stride=(2,2), with_bias=False, name='Convolution_10')

    # Convolution_9 -> 128,1,4
    h6 = PF.convolution(h4, outmaps=128, kernel=(1,1), pad=(0,0), stride=(2,2), with_bias=False, name='Convolution_9')
    # BatchNormalization_10
    h5 = PF.batch_normalization(h5, batch_stat=not test, name='BatchNormalization_10')
    # BatchNormalization_8
    h6 = PF.batch_normalization(h6, batch_stat=not test, name='BatchNormalization_8')
    # ReLU_8
    h5 = F.relu(h5, inplace=True)
    # SELU
    h6 = F.selu(h6)
    # Convolution_11
    h5 = PF.convolution(h5, outmaps=128, kernel=(5,3), pad=(2,1), with_bias=False, name='Convolution_11')
    # DepthwiseConvolution_3
    h6 = PF.depthwise_convolution(h6, pad=(2,2), name='DepthwiseConvolution_3')
    # BatchNormalization_11
    h5 = PF.batch_normalization(h5, batch_stat=not test, name='BatchNormalization_11')
    # BatchNormalization_9
    h6 = PF.batch_normalization(h6, batch_stat=not test, name='BatchNormalization_9')

    # Add2_3 -> 128,1,4
    h6 = F.add2(h6, h5, inplace=True)
    # ReLU_10
    h6 = F.relu(h6, inplace=True)
    # RepeatStart_3
    for i in range(1):

        # Convolution_13
        h7 = PF.convolution(h6, outmaps=128, pad=(1,1), with_bias=False, name='Convolution_13[' + str(i) + ']')
        # BatchNormalization_13
        h7 = PF.batch_normalization(h7, batch_stat=not test, name='BatchNormalization_13[' + str(i) + ']')
        # ReLU_11
        h7 = F.relu(h7, inplace=True)
        # Convolution_14
        h7 = PF.convolution(h7, outmaps=128, kernel=(1,3), pad=(0,1), with_bias=False, name='Convolution_14[' + str(i) + ']')
        # BatchNormalization_14
        h7 = PF.batch_normalization(h7, batch_stat=not test, name='BatchNormalization_14[' + str(i) + ']')

        # Add2_4 -> 128,1,4
        h8 = F.add2(h6, h7, inplace=False)
        # ReLU_13
        h8 = F.relu(h8, inplace=True)
        # RepeatEnd_3
        h6 = h8

    # Convolution_17 -> 256,1,2
    h9 = PF.convolution(h8, outmaps=256, pad=(1,1), stride=(2,2), with_bias=False, name='Convolution_17')

    # Convolution_16 -> 256,1,2
    h10 = PF.convolution(h8, outmaps=256, kernel=(1,1), pad=(0,0), stride=(2,2), with_bias=False, name='Convolution_16')

    # Convolution_4 -> 256,1,2
    h11 = PF.convolution(h8, outmaps=256, pad=(1,1), stride=(2,2), with_bias=False, name='Convolution_4')
    # BatchNormalization_17
    h9 = PF.batch_normalization(h9, batch_stat=not test, name='BatchNormalization_17')
    # BatchNormalization_12
    h10 = PF.batch_normalization(h10, batch_stat=not test, name='BatchNormalization_12')
    # BatchNormalization_19
    h11 = PF.batch_normalization(h11, batch_stat=not test, name='BatchNormalization_19')
    # ReLU_14
    h9 = F.relu(h9, inplace=True)
    # PReLU_2
    h10 = PF.prelu(h10, base_axis=1, shared=False, name='PReLU_2')
    # ReLU_3
    h11 = F.relu(h11, inplace=True)
    # DepthwiseConvolution_4
    h10 = PF.depthwise_convolution(h10, pad=(2,2), name='DepthwiseConvolution_4')
    # BatchNormalization_16
    h10 = PF.batch_normalization(h10, batch_stat=not test, name='BatchNormalization_16')

    # Add2_6 -> 256,1,2
    h9 = F.add2(h9, h11, inplace=False)
    # Convolution_18
    h9 = PF.convolution(h9, outmaps=256, pad=(1,1), with_bias=False, name='Convolution_18')
    # BatchNormalization_18
    h9 = PF.batch_normalization(h9, batch_stat=not test, name='BatchNormalization_18')

    # Add2_5 -> 256,1,2
    h10 = F.add2(h10, h9, inplace=True)
    # ReLU_16
    h10 = F.relu(h10, inplace=True)
    # RepeatStart_4
    for i in range(1):
        # ReLU_19
        h10 = F.relu(h10, inplace=False)
        # RepeatEnd_4

    # Convolution_24 -> 512,1,1
    h12 = PF.convolution(h10, outmaps=512, kernel=(1,1), pad=(0,0), stride=(2,2), with_bias=False, name='Convolution_24')

    # Convolution_23 -> 512,1,1
    h13 = PF.convolution(h10, outmaps=512, kernel=(1,1), pad=(0,0), stride=(2,2), with_bias=False, name='Convolution_23')
    # BatchNormalization_24
    h12 = PF.batch_normalization(h12, batch_stat=not test, name='BatchNormalization_24')
    # BatchNormalization_23
    h13 = PF.batch_normalization(h13, batch_stat=not test, name='BatchNormalization_23')
    # Swish
    h12 = F.swish(h12)
    # Convolution_25
    h12 = PF.convolution(h12, outmaps=512, kernel=(1,3), pad=(0,1), with_bias=False, name='Convolution_25')
    # BatchNormalization_25
    h12 = PF.batch_normalization(h12, batch_stat=not test, name='BatchNormalization_25')

    # Add2_8 -> 512,1,1
    h13 = F.add2(h13, h12, inplace=True)
    # ReLU_22
    h13 = F.relu(h13, inplace=True)
    # RepeatStart_5
    for i in range(1):

        # Convolution_27 -> 133,1,1
        h14 = PF.convolution(h13, outmaps=133, pad=(1,1), with_bias=False, name='Convolution_27[' + str(i) + ']')
        # BatchNormalization_5
        h14 = PF.batch_normalization(h14, batch_stat=not test, name='BatchNormalization_5[' + str(i) + ']')
        # PReLU
        h14 = PF.prelu(h14, base_axis=1, shared=False, name='PReLU[' + str(i) + ']')
        # DepthwiseConvolution_2
        h14 = PF.depthwise_convolution(h14, pad=(2,2), name='DepthwiseConvolution_2[' + str(i) + ']')
        # BatchNormalization
        h14 = PF.batch_normalization(h14, batch_stat=not test, name='BatchNormalization[' + str(i) + ']')
        # ReLU
        h14 = F.relu(h14, inplace=True)
        # DepthwiseConvolution
        h14 = PF.depthwise_convolution(h14, pad=(2,2), name='DepthwiseConvolution[' + str(i) + ']')
        # BatchNormalization_27
        h14 = PF.batch_normalization(h14, batch_stat=not test, name='BatchNormalization_27[' + str(i) + ']')
        # ReLU_23
        h14 = F.relu(h14, inplace=True)
        # Convolution_28 -> 512,1,1
        h14 = PF.convolution(h14, outmaps=512, pad=(1,1), with_bias=False, name='Convolution_28[' + str(i) + ']')
        # BatchNormalization_15
        h14 = PF.batch_normalization(h14, batch_stat=not test, name='BatchNormalization_15[' + str(i) + ']')
        # PReLU_3
        h14 = PF.prelu(h14, base_axis=1, shared=False, name='PReLU_3[' + str(i) + ']')
        # Convolution
        h14 = PF.convolution(h14, outmaps=h14.shape[0 + 1], pad=(1,1), name='Convolution[' + str(i) + ']')
        # BatchNormalization_28
        h14 = PF.batch_normalization(h14, batch_stat=not test, name='BatchNormalization_28[' + str(i) + ']')

        # Add2_7 -> 512,1,1
        h15 = F.add2(h13, h14, inplace=False)
        # ReLU_25
        h15 = F.relu(h15, inplace=True)
        # RepeatEnd_5
        h13 = h15
    # Affine_2 -> 2
    h15 = PF.affine(h15, n_outmaps=(2,), name='Affine_2')
    # SquaredError_2
    h15 = F.squared_error(h15, y)
    return h15

class HoleExistNetwork:
    def create(type=0):
        cur = os.path.dirname(__file__)

        with nn.parameter_scope('holeexist'):
            #nn.load_parameters(cur + '/holeexist/20230225_191153/results.nnp')

            params = [
                ['20230225_191153', network],       # 0
                ['20230215_010324', network2],      # 1
                ['aaa', network2],      # 2
                ['20230408_113927__3', network3],   # 3 100M
                ['20230408_120831__4', network4],   # 4
                ['20230412_231544__5', network5],   # 5 50M
                ['20230413_163836__6', network6],   # 6 40M
                ['20230413_221501__7', network7],   # 6 40M
                ['20230413_233034__8', network8],   # 6 40M
                ['20230414_113918__9', network9],   # 9 30M
                ['20230413_202106__10', network10],   # 10 50M
                ['20230417_171040_11', network10],   # 11 50M 10のNG判定画像の学習追加版
                ['20230912_055333_12', network11],   # 12 10M
                ['', None],
            ]

            nn.load_parameters(cur + '/holeexist/' + params[type][0] +  '/results.nnp')

            x = nn.Variable((1, 1, 480, 245))
            y = params[type][1](x, 1, True)

        return x, y


if __name__ == "__main__":
    holeexist = HoleExistNetwork()
    x, y = holeexist.create(0)
