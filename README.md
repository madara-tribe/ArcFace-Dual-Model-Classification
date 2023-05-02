# ArcFace-System
ArcFace multi-classification system with Pytorch and tensorflow




# Performance

| Model | Head | Backborn | class | accuracy |
| :---         |     :---:      |     :---:      |     :---:      |         ---: |
| ArcFace | ArcFace head| efficientnetv2_s | industry parts (=122) | 10%|
| meta-model | Linear+softmax | resnet18| color (=11)  | 88%|
| meta-model | Linear+softmax | resnet18| shape (=2)  | 97%|
| ArcFace-model + meta-model | / | / | 122  | 86%|

# validation loss curve

## ArcFace

<img width="673" alt="valid:arcloss" src="https://user-images.githubusercontent.com/48679574/235736339-6ff081d5-5c15-4cda-a344-0d3c7203c6f8.png">
## color

<img width="689" alt="valid:colorloss" src="https://user-images.githubusercontent.com/48679574/235736415-558dd327-efa8-4aa3-a264-ddd7ec52880f.png">
## shape

<img width="676" alt="valid:shape_losss" src="https://user-images.githubusercontent.com/48679574/235736439-99f855bf-d5ff-430b-bf2a-0665b2a45e41.png">
