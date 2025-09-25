# Two_Stage_PAL
This project relatee to the implementation and demo of "Deep Preprocessing Method for Speech Restoration in Parametric Array Loudspeakers via Time-Frequency Domain Modeling", which has been accepted by *IEEE SPL*. Our proposed method introduces two key extensions: the first implementation of effective speech restoration based on DNN methods using real-world PAL, and a novel two-stage strategy featuring the network-DiffVF approach, which addresses adversarial issues inherent in two-network strategies.

In this project, the primary basis is the original implementation of [TF-GridNet](https://github.com/espnet/espnet/blob/master/espnet2/enh/separator/tfgridnet_separator.py). Notably, the project only encompasses the training and inference phase. The parameters related to modeling the PAL process, including the DiffVF kernel and transducer response, are pre-identified using direct frequency-domain division.

## Pretrain Models

We release the model trained on the VCTK dataset, [there](https://github.com/MWY0615/Two_Stage_PAL/tree/main/Exp).


## Running Experiments

```shell
# Train the model.
bash train.sh
# Decode the model.
bash decode.sh
```

## Citation
If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry.

```bibtex
@article{ma2025deep,
  title={Deep Preprocessing Method for Speech Restoration in Parametric Array Loudspeakers via Time-Frequency Domain Modeling},
  journal={IEEE Signal Processing Letters},
  year={2025},
  publisher={IEEE},
  doi = {https://doi.org/10.1109/LSP.2025.3609247},
  author={Ma, Wenyao and Zhu, Yunxi and Yang, Jun}
}
```