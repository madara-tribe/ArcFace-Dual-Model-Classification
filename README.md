# ArcFace-DualModel-Classification (Pytorch and tensorflow)

ArcFace dual model classification with Pytorch and tensorflow.
This repogitory is pytorch with 2 model (ArcFace + meta-model).


<b>Dual model classification</b>

At first extract top 50 candidates by arcface-model and next classify color and shape label by meta model.

Finally, classify and get result class label.

<img src="https://user-images.githubusercontent.com/48679574/235917765-d0100cc2-b282-4497-a33b-17a88d2013b3.jpg" width="600px" height="400px"/>


## Version

```sh
・python : 3.8
・pytorch : 2.0.0+cu117
・torchvision : 0.15.1+cu117
```

## Performance

| Model | Head | Backborn | class | accuracy |
| :---         |     :---:      |     :---:      |     :---:      |         ---: |
| ArcFace | ArcFace head| efficientnetv2_s | industry parts (=122) | 10%|
| meta-model | Linear+softmax | resnet18| color (=11)  | 88%|
| meta-model | Linear+softmax | resnet18| shape (=2)  | 97%|
| ArcFace-model + meta-model | / | / | 122  | 86%|

## validation loss curve

#### ArcFace / Color / Shape
<img src="https://user-images.githubusercontent.com/48679574/235736339-6ff081d5-5c15-4cda-a344-0d3c7203c6f8.png" width="300px"><img src="https://user-images.githubusercontent.com/48679574/235736415-558dd327-efa8-4aa3-a264-ddd7ec52880f.png" width="300px"><img src="https://user-images.githubusercontent.com/48679574/235736439-99f855bf-d5ff-430b-bf2a-0665b2a45e41.png" width="300px">

## Dataset 

Industrial parts. Refer to file 'dataset/cs_label.json'
```sh
・122 class (main class to classify)
・9 color class (meta label)
・2 shape label (meta label)
```


## useful technics

#### ・image padding resize (example)

<img src="https://user-images.githubusercontent.com/48679574/147999782-4e9e84cc-09f1-4a15-994b-1a2cb1f8e8b1.jpeg" width="500px">

#### ・[Lambda Layer](https://github.com/madara-tribe/Lambda-Networks)

## Pytorch Version (This repository)
Pytorch Version repository is as follows:
```sh
$ git clone  Arcface_pytorch
```

## Tensorflow Version

Tensorflow Version repository is as follows:
```sh
$ git clone -b  ArcfaceSystem/tensorflow Arcface_tensorflow
```
