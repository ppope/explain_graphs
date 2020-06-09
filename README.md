# `explain_molecules`

Code for "Explainability methods for graph convolutional neural networks" - PE Pope*, S Kolouri*, M Rostami, CE Martin, H Hoffmann (CVPR 2019)

[paper](openaccess.thecvf.com/content_CVPR_2019/papers/Pope_Explainability_Methods_for_Graph_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)

_"With the growing use of graph convolutional neural networks (GCNNs) comes the need for explainability. In thispaper, we introduce explainability methods for GCNNs. We develop the graph analogues of three prominent explain-ability methods for convolutional neural networks: contrastive gradient-based (CG) saliency maps, Class Activation Mapping (CAM), and Excitation Backpropagation (EB) and their variants, gradient-weighted CAM (Grad-CAM)i and contrastive EB (c-EB). We show a proof-of-concept ofthese methods on classification problems in two applicationi domains: visual scene graphs and molecular graphs. Tocompare the methods, we identify three desirable propertiesi of explanations: (1) their importance to classification, asmeasured by the impact of occlusions, (2) their contrastivity with respect to different classes, and (3) their sparseness on a graph. We call the corresponding quantitative metrics fidelity, contrastivity, and sparsity and evaluate them for each method. Lastly, we analyze the salient subgraphs obtained from explanations and report frequently occurring patterns."_

This repository replicates results on molecular datasets.

### Environment

#### `docker`

Start `deepchem` container
```
docker run \
   --gpus 0 \
   --name=explain_molecules \
   -i -t -p 8888:8888 \
   --volume "$PWD:/home" \
   deepchemio/deepchem:2.1.0
```

Setup container environment:
```
pip install requirements.txt 
```

To use notebooks: 
```
cd home; jupyter notebook --ip=0.0.0.0 --allow-root
```

Then navigate to localhost:8888 and run the notebooks

#### Data

Download datasets and place them in 'data/':

* BBBP: [http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/BBBP.csv](http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/BBBP.csv)
* BACE: [http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/bace.csv](http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/bace.csv)
* Tox21: [http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/tox21.csv.gz](http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/tox21.csv.gz)

#### Run

Run the following `notebooks` to replicate the results

Train and evaluate graph convolutional neural networks (GCN)
```
0.1-gcn-train-eval.ipynb
```

Explain molecular classifications and visualize:
```
0.2-explain-viz-save-masks.ipynb
```

Analyze results for reoccurring substructures
```
0.3-substructure-search.ipynb
```

Calculate fidelity metric on results
```
0.4-occlusion.ipynb
```

### Citation

Please cite our work if you find it useful:

```
 @InProceedings{Pope_2019_CVPR,
author = {Pope, Phillip E. and Kolouri, Soheil and Rostami, Mohammad and Martin, Charles E. and Hoffmann, Heiko},
title = {Explainability Methods for Graph Convolutional Neural Networks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
