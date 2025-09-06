# Two_Stage_PAL
This project relatee to the implementation and demo of "Deep Preprocessing Method for Speech Restoration in Parametric Array Loudspeakers via Time-Frequency Domain Modeling", which has been accepted by *IEEE SPL*. Our proposed method introduces two key extensions: the first implementation of effective speech restoration based on DNN methods using real-world PAL, and a novel two-stage strategy featuring the network-DiffVF approach, which addresses adversarial issues inherent in two-network strategies.

In this project, the primary basis is the original implementation of [TF-GridNet](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tfgridnet_separator.py). Notably, the project only encompasses the training and inference phase. The parameters related to modeling the PAL process, including the DiffVF kernel and transducer response, are pre-identified using direct frequency-domain division.

## Running Experiments

```shell
# Train the model.
bash train.sh
# Decode the model.
bash decode.sh
```

## Citation