# PAPERCLIP: Associating Astronomical Observations and Natural Language with Multi-Modal Models<!-- omit from toc -->

[Siddharth Mishra-Sharma](mailto:smsharma@mit.edu), Yiding Song, and Jesse Thaler

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2403.08851%20-green.svg)](https://arxiv.org/abs/2403.08851)
[![draft](https://img.shields.io/badge/Draft-PDF-blue)](https://github.com/smsharma/HubbleCLIP/blob/main-pdf/paper/hubble_paperclip.pdf)

![Figure](paper/plots/figure.png)

## Contents<!-- omit from toc -->

- [Abstract](#abstract)
- [Paper draft](#paper-draft)
- [Requirements](#requirements)
- [Code overview](#code-overview)
- [Fine-tuned CLIP model and _Hubble_ data](#fine-tuned-clip-model-and-hubble-data)
- [Citation](#citation)

## Abstract

We present PAPERCLIP (Proposal Abstracts Provide an Effective Representation for Contrastive Language-Image Pre-training), a method which associates astronomical observations imaged by telescopes with natural language using a neural network model. The model is fine-tuned from a pre-trained Contrastive Language-Image Pre-training (CLIP) model using successful observing proposal abstracts and corresponding downstream observations, with the abstracts optionally summarized via guided generation using large language models (LLMs). Using observations from the Hubble Space Telescope (HST) as an example, we show that the fine-tuned model embodies a meaningful joint representation between observations and natural language through tests targeting image retrieval (i.e., finding the most relevant observations using natural language queries) and description retrieval (i.e., querying for astrophysical object classes and use cases most relevant to a given observation). Our study demonstrates the potential for using generalist foundation models rather than task-specific models for interacting with astronomical data by leveraging text as an interface.

## Paper draft

[Link to paper draft](https://github.com/smsharma/HubbleCLIP/blob/main-pdf/paper/hubble_paperclip.pdf). The PDF is compiled automatically from the `main` branch into the `main-pdf` branch on push.

## Requirements

Since PyTorch and Jax can be [tricky to have under the same roof](https://github.com/google/jax/issues/18032), the Python environment for downloading data and guided LLM summarization using `Outlines` is defined in `environment_outlines.yml`, and the one for training and evaluating the CLIP model in `environment.py`. To create the environment run e.g.,
``` sh
mamba env create --file environment.yaml
```

## Code overview

The code primarily uses Jax. The main components are:

- The script for downloading the data is [download_data.py](download_data.py), the summarization script is [summarize.py](summarize.py), and training script is [train.py](train.py).
- [notebooks/01_create_dataset.ipynb](notebooks/01_create_dataset.ipynb) is used to create the `tfrecords` data used for training.
- [notebooks/03_eval.ipynb](notebooks/03_eval.ipynb) creates the qualitative and quantitative evaluation plots.
- [notebooks/09_dot_product_eval.ipynb](notebooks/09_dot_product_eval.ipynb) generates additional quantitative evaluation.

## Fine-tuned CLIP model and _Hubble_ data

_Coming soon._

## Citation

If you use this code, please cite our paper:

```
@misc{mishrasharma2024paperclip,
      title={PAPERCLIP: Associating Astronomical Observations and Natural Language with Multi-Modal Models}, 
      author={Siddharth Mishra-Sharma and Yiding Song and Jesse Thaler},
      year={2024},
      eprint={2403.08851},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM}
}
```