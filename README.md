# MDNet: Multi-Patch Dense Network
A Keras implmentation of the MDNet, published in the 2018 paper titled, ["Multi-Patch Dense Network for Coral Classification"](https://afrl.cse.sc.edu/afrl/publications/public_html/papers/ModasshirOceans2018.pdf). The network was originally designed for the purpose of coral reef patch-based image classification, and currently is the state-of-the-art for the Moorea Labeled Coral (MLC) dataset, originally published by Beijbom et al. 2012.

This custom architecture learns class categories at multiple scales and adopts the use of densely connected convolutional layers to reduce overfitting. MDNet extracts features from image-patches of different sizes in parallel, later concatenating them together to create a final descriptor for each annotated point. This technique allows for training  end-to-end, learning information at different scales without having to perform costly resizing operations on each patch as done in previous works.

![](Paper_Figures/MDNet_Arch.PNG)

## Code
```python
from MDNet import *

n_classes = 9

model = build_MDNet(9, (112, 112, 3),
                    num_pipelines = 4,
                    num_blocks = 3,
                    num_layers = 5, 
                    num_filters = 12,
                    dropout_rate = .75,
                    decrease_by = .25)
                     
model.compile()

```

