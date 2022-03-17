# Mucko
implementation for Mucko: Multi-Layer Cross-Modal Knowledge Reasoning for Fact-based Visual Question Answering

![image](https://github.com/AndersonStra/Mucko/blob/main/mucko.PNG)


## Requirements
pytorch == 1.2.0                                        
dgl == 0.4.3


## Train
```
python train.py --dataset okvqa --validate --save-dirpath save_dir --gpu-ids 0
```

## Reference
```
@inproceedings{zhu1020mucko,
  author    = {Zihao Zhu and
               Jing Yu and
               Yujing Wang and
               Yajing Sun and
               Yue Hu and
               Qi Wu},
  title     = {Mucko: Multi-Layer Cross-Modal Knowledge Reasoning for Fact-based
               Visual Question Answering},
  booktitle = {Proceedings of the International Joint Conference on Artificial Intelligence},
  pages     = {1097--1103},
  year      = {2020}
}
```
