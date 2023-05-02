# ArcFace-System (Pytorch and tensorflow)

ArcFace multi-classification system with Pytorch and tensorflow


# Abstract (This repository is Pytorch Version)

This is pytorch version ArcFace system classification with 2 model (ArcFace + meta-model)


### Tensorflow Version

Tensorflow Version repository is as follows:
```sh
$ git clone -b  ArcfaceSystem/tensorflow
```




# Performance

| Model | Head | Backborn | class | accuracy |
| :---         |     :---:      |     :---:      |     :---:      |         ---: |
| ArcFace | ArcFace head| efficientnetv2_s | industry parts (=122) | 10%|
| meta-model | Linear+softmax | resnet18| color (=11)  | 88%|
| meta-model | Linear+softmax | resnet18| shape (=2)  | 97%|
| ArcFace-model + meta-model | / | / | 122  | 86%|

# validation loss curve

## ArcFace / Color / Shape
<img src="https://user-images.githubusercontent.com/48679574/235736339-6ff081d5-5c15-4cda-a344-0d3c7203c6f8.png" width="300px"><img src="https://user-images.githubusercontent.com/48679574/235736415-558dd327-efa8-4aa3-a264-ddd7ec52880f.png" width="300px"><img src="https://user-images.githubusercontent.com/48679574/235736439-99f855bf-d5ff-430b-bf2a-0665b2a45e41.png" width="300px">


# useful technics

### ・image padding resize (example)

<img src="https://user-images.githubusercontent.com/48679574/147999782-4e9e84cc-09f1-4a15-994b-1a2cb1f8e8b1.jpeg" width="500px">

