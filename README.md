# U-Bench: A Comprehensive Understanding of U-Net through 100-Variant Benchmarking

### Abstract

Over the past decade, U-Net has been the dominant architecture in medical image segmentation, leading to the development of thousands of U-shaped variants. Despite its widespread adoption, a comprehensive benchmark to systematically assess the performance and utility of these models is lacking, primarily due to insufficient statistical validation and limited attention to efficiency and generalization across diverse datasets. To address this gap, we present U-Bench, the first large-scale, statistically rigorous benchmark that evaluates 100 U-Net variants across 28 datasets and 10 imaging modalities. Our contributions are threefold: (1) Comprehensive Evaluation: U-Bench evaluates models along three key dimensions: statistical robustness, zero-shot generalization, and computational efficiency. We introduce a novel metric, U-Score, which jointly captures the performance-efficiency trade-off, offering a deployment-oriented perspective on model progress. (2) Systematic Analysis and Model Selection Guidance: We summarize key findings from the large-scale evaluation and systematically analyze the impact of dataset characteristics and architectural paradigms on model performance. Based on these insights, we propose a model advisor agent to guide researchers in selecting the most suitable models for specific datasets and tasks. (3) Public Availability: We provide all code, models, protocols, and weights, enabling the community to reproduce our results and extend the benchmark with future methods. In summary, U-Bench not only exposes gaps in previous evaluations but also establishes a foundation for fair, reproducible, and practically relevant benchmarking in the next decade of U-Net-based segmentation models.

### Quick Start

#### 1. Datasets

Please put the dataset (e.g. BUSI) or your own dataset as the following architecture:

```
└── U-Bench
    ├── data
        ├── busi
            ├── images
            |   ├── benign (10).png
            │   ├── malignant (17).png
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── benign (10).png
                |   ├── malignant (17).png
                |   ├── ...
        ├── your dataset
            ├── images
            |   ├── 0a7e06.png
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── ...
    ├── dataloader
    ├── models
    ├── utils
    ├── script
    ├── main.py
    └── main_multi3d.py
```

#### 2. Training & Validation

Please refer script.

We have **removed all sensitive URLs** from the repository to comply with anonymity and privacy rules.  
The **pre-trained weights, datasets, and benchmark results** will be released **after the acceptance of our paper**.

update