# ArcFace-System
ArcFace multi-classification system with Pytorch and tensorflow




# Performance

| Model | Head | Backborn | class | accuracy |
| :---         |     :---:      |     :---:      |     :---:      |         ---: |
| ArcFace | ArcFace head| efficientnetv2_s | 122 | 10%|
| meta-model | Linear+softmax | resnet18| color (=11)  | 88%|
| meta-model | Linear+softmax | resnet18| shape (=2)  | 97%|
| ArcFace-model + meta-model | / | / | 122  | 86%|
