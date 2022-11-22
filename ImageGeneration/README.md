To set up the environment, run
```
pip install -r requirements.txt
```

The configuration files are in ```./configs/mixup_flow/```


To train a model, run
```
python ./main.py --config ./configs/mixup_flow/imagenet_pytorch_mixup_gaussian.py --eval_folder eval --mode train --workdir ./logs/YOUR_LOG_PATH
```
