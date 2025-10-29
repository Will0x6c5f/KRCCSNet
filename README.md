PyTorch implementation of KR-CCSNet.
We use accelerate as our distribute training framwork.
Please install dependencies in requirements.txt before running any scripts.
Training and fine-tuning requires around 40G GPU memory total for biggest model.
You can change the distribute training setting in acc_config.yaml 
Testing can be done with single GPU (even CPU, but need slightly modify the code)

Please prepare dataset in the following structure: 
```
code
├── BSDS500
│   ├── set14
│   │   ├── set14
│   │   │   ├── xxx.png
│   │   │   ...
│   ├── set5
│   │   ├── set5
│   │   │   ├── xxx.png 
│   │   │   ...
│   ├── train
│   │   ├── train
│   │   │   ├── xxx.png
│   │   │   ...
│   ├── val
│   │   ├── val
│   │   │   ├── xxx.png
│   │   │   ...
```

Our model is in model/krccsnet.py


QuickStart:

train:
bash tools/dtrain.sh krccsnet_train 0.25 0123.yaml 

reparameterize:
run reparam.ipynb

fine-tune:
bash tools/dfinetune.sh krccsnet 0.25 0123.yaml 

test:
#test krccsnet
python test.py --model krccsnet --sensing-rate 0.25 
#test krccsnet_train(i.e. LKSN+ARM )
python test.py --model krccsnet_train --sensing-rate 0.25 

