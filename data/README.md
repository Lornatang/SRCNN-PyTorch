# Usage

## Download train datasets

### T91

- Image format
    - [Google Driver](https://drive.google.com/drive/folders/1PYizfnKq-UtRCDoSy79PGA4FC5HqAqch?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1M0u-BPTdokxO452j7vxW4Q) access: `llot`

- LMDB format
    - [Google Driver](https://drive.google.com/drive/folders/1PYizfnKq-UtRCDoSy79PGA4FC5HqAqch?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1D-OmhMCCFHuvZ_7OugDHWQ) access: `llot`

## Download valid dataset

### Set5

- Image format
    - [Google Driver](https://drive.google.com/file/d/1GJZztdiJ6oBmJe9Ntyyos_psMzM8KY4P/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1_B97Ga6thSi5h43Wuqyw0Q) access:`llot`

### Set14

- Image format
    - [Google Driver](https://drive.google.com/file/d/14bxrGB3Nej8vBqxLoqerGX2dhChQKJoa/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1wy_kf4Kkj2nSkgRUkaLzVA) access:`llot`

### BSD100

- Image format
    - [Google Driver](https://drive.google.com/file/d/1xkjWJGZgwWjDZZFN6KWlNMvHXmRORvdG/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1EBVulUpsQrDmZfqnm4jOZw) access:`llot`

## Train dataset struct information

### Image format

```text
- T91
    - X2
        - train
            - inputs
                - ...
            - target
                - ...
        - valid
            - inputs
                - ...
            - target
                - ...
    - X4
        - train
            - inputs
                - ...
            - target
                - ...
        - valid
            - inputs
                - ...
            - target
                - ...
    - ...
```

### LMDB format

```text
- train_lmdb
    - LR
        - T91_X4_lmdb
            - data.mdb
            - lock.mdb
    - HR
        - T91_X4_lmdb
            - data.mdb
            - lock.mdb
- valid_lmdb
    - LR
        - T91_X4_lmdb
            - data.mdb
            - lock.mdb
    - HR
        - T91_X4_lmdb
            - data.mdb
            - lock.mdb
```

## Test dataset struct information

### Image format

```text
- Set5
    - GTmod12
        - baby.png
        - bird.png
        - ...
    - LRbicx4
        - baby.png
        - bird.png
        - ...
- Set14
    - GTmod12
        - baboon.png
        - barbara.png
        - ...
    - LRbicx4
        - baboon.png
        - barbara.png
        - ...
```
