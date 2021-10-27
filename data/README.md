# Usage

## Download train datasets

### T91

- Image format
    - [Google Driver](https://drive.google.com/drive/folders/1PYizfnKq-UtRCDoSy79PGA4FC5HqAqch?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1Oa1oas0GOT78DX1IAX7svg) access: `llot`

- LMDB format
    - [Google Driver](https://drive.google.com/drive/folders/1PYizfnKq-UtRCDoSy79PGA4FC5HqAqch?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1Oa1oas0GOT78DX1IAX7svg) access: `llot`

## Download val dataset

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
        - traina
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
    - HR
        - T91
            - data.mdb
            - lock.mdb
    - LRbicx4
        - T91
            - data.mdb
            - lock.mdb
- valid_lmdb
    - HR
        - T91
            - data.mdb
            - lock.mdb
    - LRbicx4
        - T91
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
