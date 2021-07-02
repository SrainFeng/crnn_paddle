import paddle.optimizer as optim
import time
from pathlib import Path
import os
import paddle


def get_optimizer(config, model, scheduler=None):
    """
    根据配置文件获取相应的优化器
    :param scheduler: 学习率策略
    :param config: 配置文件
    :param model: 模型文件
    :return: 优化器
    """

    optimizer = None

    if scheduler is None:
        scheduler = config.TRAIN.LR

    if config.TRAIN.OPTIMIZER == "sgd":
        optimizer = optim.SGD(
            learning_rate=scheduler,
            parameters=filter(lambda p: bool(1 - p.stop_gradient), model.parameters()),
            weight_decay=config.TRAIN.WD
        )
    elif config.TRAIN.OPTIMIZER == "adam":
        optimizer = optim.Adam(
            learning_rate=scheduler,
            parameters=filter(lambda p: bool(1 - p.stop_gradient), model.parameters())
        )
    elif config.TRAIN.OPTIMIZER == "rmsprop":
        optimizer = optim.RMSProp(
            learning_rate=scheduler,
            momentum=config.TRAIN.MOMENTUM,
            parameters=filter(lambda p: bool(1 - p.stop_gradient), model.parameters()),
            weight_decay=config.TRAIN.WD
        )

    return optimizer


def create_log_folder(cfg, phase='train'):
    """
    创建输出日志的文件夹
    :param cfg:
    :param phase:
    :return:
    """
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME

    time_str = time.strftime('%Y-%m-%d-%H-%M')

    checkpoints_output_dir = root_output_dir / dataset / model / time_str / 'checkpoints'
    print('=> creating {}'.format(checkpoints_output_dir))
    checkpoints_output_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_log_dir = root_output_dir / dataset / model / time_str / 'log'
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return {'chs_dir': str(checkpoints_output_dir), 'tb_dir': str(tensorboard_log_dir)}


def get_batch_label(d, i):
    label = []
    for idx in i:
        label.append(list(d.labels[idx].values())[0])
    return label


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        length = []
        result = []
        decode_flag = True if type(text[0]) == bytes else False

        for item in text:
            text_code = []
            if decode_flag:
                item = item.decode('utf-8', 'strict')
            length.append(len(item))
            for char in item:
                index = self.dict[char]
                text_code.append(index)
            result.append(text_code)

        max_len = max(length)

        for i in range(len(result)):
            if len(result[i]) < max_len:
                result[i] = self._fill_list(result[i], max_len)

        text = result

        return paddle.to_tensor(text, dtype="int32"), paddle.to_tensor(length, "int64")

    def _fill_list(self, my_list: list, length, fill=0):  # 使用 fill字符/数字 填充，使得最后的长度为 length
        if len(my_list) >= length:
            return my_list
        else:
            return my_list + (length - len(my_list)) * [fill]


    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], paddle.to_tensor([l], dtype="int32"), raw=raw))
                index += l
            return texts


def get_char_dict(path):
    with open(path, 'rb') as file:
        char_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}


def model_info(model):  # Plots a line-by-line description of a  model
    global i
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if bool(1-x.stop_gradient))  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, bool(1-p.stop_gradient), p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


from lib.models.crnn import CRNN
if __name__ == '__main__':
    crnn = CRNN(32, 1, 1, 256)
    model_info(crnn)
