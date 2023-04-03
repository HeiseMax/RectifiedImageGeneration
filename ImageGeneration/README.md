To set up the environment, run
```
pip install -r requirements.txt
```

The configuration files are in ```./configs/mixup_flow/```


To train a model, run
```
python ./main.py --config ./configs/mixup_flow/imagenet_pytorch_mixup_gaussian.py --eval_folder eval --mode train --workdir ./logs/YOUR_LOG_PATH
```


tensorboard --logdir logs/fit

python ./main.py --config ./configs/mixup_flow/mnist_mixup_gaussian_from_vp_ddpm.py --eval_folder eval --mode train --workdir ./logs/test2 --config.model.ema_rate 0.999999 --config.model.dropout 0.15