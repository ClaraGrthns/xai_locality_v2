# Analyzing Local Fidelity of XAI Methods for Tabular Data

## Abstract
Explainable AI aims to make black-box model decisions understandable, with local explanations being a common approach for interpreting individual predictions. Local fidelity, which is the alignment between an explanation and the black-box model in the explained instance's neighborhood, is an important property of such explanations. This study investigates local fidelity of local explanations by evaluating common explanation methods on regression and classification tasks. We contextualize local fidelity with the complexity of the underlying black-box model. Additionally, we assess whether local explanations provide meaningful value over trivial baseline approaches. Furthermore, we analyze neighborhood sizes in which explanations remain accurate. Our results show a significant divergence: local fidelity is high only for simple models, where explanations may be unnecessary. Conversely, for complex models, where interpretability is essential, local explanations mostly fail to accurately capture model behavior. For classification tasks, local explanations often provide limited additional insights into model behavior within small neighborhoods around individual predictions. While absolute local fidelity values vary by method and dataset, we consistently find that explanations remain accurate only in very small neighborhoods. These findings hold significant implications for practitioners and end-users, suggesting that local explanations may offer limited value for understanding complex model behavior beyond the explained instance.

### XAI methods
- LIME (default binary and continuos)
- Saliency Maps
- Integrated Gradients
- SmoothGrad + Integrated Gradients
### Datasets

#### Standard Datasets
[TabularBenchmark](https://pytorch-frame.readthedocs.io/en/latest/generated/torch_frame.datasets.TabularBenchmark.html#torch_frame.datasets.TabularBenchmark)

#### Synthetic Datasets
- Using sklearns method: ```sklearn.datasets.make_classification```
[Link to dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification)
- Custom Regression Synthetic Data ```make_custom_regression_data```

### Models

#### Deep Learning Models (PyTorch-Frame)
- [ResNet (Gorishniy et al., 2021)](https://github.com/yandex-research/rtdl-revisiting-models)
- [ExcelFormer (Chen et al., 2023a)](https://github.com/WhatAShot/ExcelFormer)
- [Trompt (Chen, et al., 22023)](https://arxiv.org/abs/2305.18446)
- [FTTransformer (Gorishniy et al., 2021)](https://github.com/yandex-research/rtdl-revisiting-models)
- [TabNet (Arik Sercan O., 2021)](https://github.com/dreamquark-ai/tabnet)
- [TabTransformer (Huang et al., 2020)](https://github.com/lucidrains/tab-transformer-pytorch)
- Simple MLP

#### Gradient Boosting Models
- XGBoost
- LightGBM


## Attribution
This repository contains code adapted from the python package [PyTorch Frame (PyG-team)](https://github.com/pyg-team/pytorch_geometric).  
- Original source: [GitHub link to original script](https://github.com/pyg-team/pytorch-frame/benchmark/data_frame_benchmark.py)  
- License: MIT ([link](https://github.com/pyg-team/pytorch_geometric/pytorch-frame/LICENSE))  
Modifications include dataset adaptation for our specific use case.
