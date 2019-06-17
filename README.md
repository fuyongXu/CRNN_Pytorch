# CRNN_Pytorch(Convolutional Recurrent Neural Network)
The implements of the Convolutional Recurrent Neural Network(CRNN)in Pytorch.Origin repo could be found in [crnn](https://github.com/bgshih/crnn)  

## Requirements  
 * Pytorch >= 0.4  
 * lmdb
 * torchvision  

## demo  
A demo program can be found in **demo.py**. launch the demo by:    

```python
python demo.py
```

## Train a new model

Construct dataset following [(origin guide)](https://github.com/bgshih/crnn#train-a-new-model). If you want to train with variable length images (keep the origin ratio for example), please modify the tool/create_dataset.py and sort the image according to the text length.
Execute python train.py --adadelta --trainRoot {train_path} --valRoot {val_path} --cuda. Explore train.py for details.

Feel free to contact me if you have any suggestions or questions, issues are welcome, create a PR if you find any bugs or you want to contribute.😄
