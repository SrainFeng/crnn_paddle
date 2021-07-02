# crnn_paddle
CRNN Chinese text recognition implement with paddle
## Environments
1. **paddle2.1** with cuda 10.2
2. yaml
3. easydict
## Data
### Synthetic Chinese String Dataset
1. Download the [dataset](https://pan.baidu.com/s/1ufYbnZAZ1q0AlK7yZ08cvQ)
2. Edit **lib/config/360CC_config.yaml** DATA:ROOT to you image path

```angular2html
    DATASET:
      ROOT: 'to/your/images/path'
3. Download the [labels](https://pan.baidu.com/s/1oOKFDt7t0Wg6ew2uZUN9xg) (password: eaqb)
4. Put *char_std_5990.txt* in **lib/dataset/txt/**
5. And put *train.txt* and *test.txt* in **lib/dataset/txt/**

    eg. test.txt
```
    20456343_4045240981.jpg 89 201 241 178 19 94 19 22 26 656
    20457281_3395886438.jpg 120 1061 2 376 78 249 272 272 120 1061
    ...


### Or your own data
1. Edit **lib/config/OWN_config.yaml** DATA:ROOT to you image path
```angular2html
    DATASET:
      ROOT: 'to/your/images/path'
```
2. And put your *train_own.txt* and *test_own.txt* in **lib/dataset/txt/**

    eg. test_own.txt
```
    img_name.jpg 你好啊！祖国！
    ...
```
## Train
```angular2html
   [run] CUDA_VISIBLE_DEVICES=0 python train.py --cfg lib/config/360CC_config.yaml
or [run] CUDA_VISIBLE_DEVICES=0 python train.py --cfg lib/config/OWN_config.yaml
```

## Demo
```angular2html
   [run] CUDA_VISIBLE_DEVICES=0 python demo.py --image_path images/test.png --checkpoint to/your/checkpoints/path
```

## Eval Images
```angular2html
   [run] CUDA_VISIBLE_DEVICES=0 python val.py --checkpoint to/your/checkpoints/path --val_dataset to/your/images/path
```

## References
- https://github.com/Sierkinhane/crnn_chinese_characters_rec
- https://github.com/meijieru/crnn.pytorch
