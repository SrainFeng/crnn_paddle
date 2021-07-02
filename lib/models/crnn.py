import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

class BidirectionalLSTM(nn.Layer):
    # Inputs hidden units Out
    def __init__(self, nIn, nHiddent, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHiddent, direction="bidirectional", time_major=True)
        self.embedding = nn.Linear(nHiddent * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        # 如果time_major为False(默认为False)，Tensor的形状为[batch_size, time_steps, num_directions * hidden_size]
        T, b, h = recurrent.shape
        t_rec = paddle.reshape(recurrent, [T * b, h])

        output = self.embedding(t_rec)
        output = paddle.reshape(output, [T, b, -1])

        return output


class CRNN(nn.Layer):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batuchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_sublayer('conv{0}'.format(i),
                             nn.Conv2D(nIn, nOut, ks[i], ss[i], ps[i]))

            if batuchNormalization:
                cnn.add_sublayer('batchnorm{0}'.format(i), nn.BatchNorm2D(nOut))
            if leakyRelu:
                cnn.add_sublayer('relu{0}'.format(i),
                                 nn.LeakyReLU(0.2))
            else:
                cnn.add_sublayer('relu{0}'.format(i), nn.ReLU())

        convRelu(0)
        cnn.add_sublayer('pooling{0}'.format(0), nn.MaxPool2D(2, 2))
        convRelu(1)
        cnn.add_sublayer('pooling{0}'.format(1), nn.MaxPool2D(2, 2))
        convRelu(2, batuchNormalization=True)
        convRelu(3)
        cnn.add_sublayer('pooling{0}'.format(2),
                         nn.MaxPool2D((2, 2), (2, 1), (0, 1)))
        convRelu(4, batuchNormalization=True)
        convRelu(5)
        cnn.add_sublayer('pooling{0}'.format(3),
                         nn.MaxPool2D((2, 2), (2, 1), (0, 1)))
        convRelu(6, batuchNormalization=True)

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

        # 初始化参数
        for m in self.sublayers():
            self._weights_init(m)


    def forward(self, input):

        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.shape
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        # pytorch: permute
        conv = paddle.transpose(conv, perm=[2, 0, 1])
        # output = F.log_softmax(self.rnn(conv), axis=2)
        output = self.rnn(conv)

        return output

    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            v = np.random.normal(loc=0.0, scale=0.02, size=m.weight.shape).astype('float32')
            m.weight.set_value(v)
            print(classname, "init")
        elif classname.find('BatchNorm') != -1:
            v = np.random.normal(loc=0.0, scale=0.02, size=m.weight.shape).astype('float32')
            m.weight.set_value(v)
            m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))
            print(classname, "init")


def get_crnn(config):
    model = CRNN(config.MODEL.IMAGE_SIZE.H, 1, config.MODEL.NUM_CLASSES + 1, config.MODEL.NUM_HIDDEN)
    return model


def get_val_crnn(img_h, n_class, n_h, nc=1):
    model = CRNN(img_h, nc, n_class + 1, n_h)
    return model


if __name__ == '__main__':
    crnn = CRNN(32, 1, 1, 256)
    print(crnn)
    # for m in crnn.sublayers():
    #     classname = m.__class__.__name__
    #     print(classname)
        # if classname.find('Conv') != -1:
        #     print("This is conv")
        # elif classname.find('BatchNorm') != -1:
        #     print("This is BatchNorm")





