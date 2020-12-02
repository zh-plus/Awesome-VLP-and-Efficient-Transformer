## Vision Language Models

### Vision-Language Pretraining

#####  ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks

NeurIPS 2019 [[paper]](https://arxiv.org/abs/1908.02265) [[code]](https://github.com/jiasenlu/vilbert_beta) *Facebook AI Research*

   - Architecture: Two stream :arrows_clockwise: **co-attentional transformer layers**
     <img src="images\ViLBERT.png" style="zoom:20%;" />
   - Pretrain dataset: Conceptual Captions (~3.3M)
   - Pretrain Tasks
     <img src="images\ViLBERT_pretrain.png" alt="ViLBERT_pretrain" style="zoom:5%;" />
     - predicting the semantics of masked words and image regions given the unmasked inputs (Masked Multi-modal Modelling)
       **image**: Predict the semantic classes distribution using image  input/output with detection model, then minimize KL divergence between these two distributions.
       text: Same as BERT.
     - predicting whether an image and text segment correspond (Multi-modal Alignment) **with [IMG] and [CLS] output**
   - Image feature (Fast R-CNN)
     - \<image coordinates (4), area fraction, visual feature\> from pretrained object detection network
     - projected to match the visual feature
   - Text feature
      Google's WordPiece tokenizer



##### LXMERT: Learning Cross-Modality Encoder Representations from Transformers

EMNLP 2019 [[paper]](https://arxiv.org/abs/1908.07490) [[code]](https://github.com/airsplay/lxmert) *The University of North Carolina*

   - Architecture: Two stream --- Object relationship encoder (**Image**), language encoder (**Text**), cross-modality encoder.
     <img src="images\LXMERT.png" style="zoom:43%;" />
   - Pretrain dataset: COCO + Visual Genome (9.18 M)
   - Pretrain Tasks
     <img src="images\LXMERT_pretrain.png" style="zoom:43%;" />
     - MLM, Masked Object Prediction (MOP) [**feature regression** and **label classification**], Cross-modality Matching **with only [CLS] output**, Image Question Answering
   - Image feature (Fast R-CNN)
     - \<bounding box coordinates, 2048-d region-of-interest\>
     - projection
       <img src="images\LXMERT_image_feature.png" style="zoom:43%;" />
   - Text feature
     <img src="images\LXMERT_text_feature.png" style="zoom:43%;" />



##### VL-BERT: Pre-training of Generic Visual-Linguistic Representations

ICLR 2020 [[paper]]((https://arxiv.org/abs/1908.08530)) [[code]](https://github.com/jackroos/VL-BERT) *USTC & MSRA*

   - Architecture: Single stream BERT
     <img src="images\VLBert.png" style="zoom:43%;" />

   - Pretrain dataset: Conceptual Captions (3.3M) for visual-linguistic & BooksCorpus, English Wikipedia for pure text corpus

   - Pretrain Tasks

     - MLM, Masked RoI Classification with Linguistic Clues

   - Features

     - Visual Feature Embedding (Fast R-CNN)
       - visual appearance embedding: 2048-d feature
         For **Non-visual elements**, they're obtained by RoI covering the whole input image.
       - visual geometry embedding: $(\frac{x_{LT}}{W}, \frac{y_{LT}}{H}, \frac{x_{RB}}{W}, \frac{y_{RB}}{H})$ -> 2048-d representation by computing sine and cosine of different wavelengths according to "Relation networks for object detection"

     - Token Embedding
       - WordPiece Embedding
         For **Visual elements**, a special [IMG] is assigned.
     - Segment Embedding: Learnable
     - Sequence Position Embedding: Learnable



#####  VisualBERT: A Simple and Performant Baseline for Vision and Language

arXiv 2019/08, ACL 2020 [[paper]](https://arxiv.org/abs/1908.03557)  [[code]](https://github.com/uclanlp/visualbert)

   - Architecture: Single stream BERT
     <img src="images\VisualBERT.png" style="zoom:43%;" />
   - Pretrain dataset: COCO (100k)
   - Pretrain tasks:
     - Task-Agnostic Pretraining
       - MLM with only text masked
       - Sentence-image matching (Cross-modality Matching) **with only [CLS] output**
     - Task-Specific Pretraining
       using MLM with task-specific dataset, which help adapting to the new target domain.
   - Features
     - Image feature (Fast R-CNN)
       visual feature representation: bounding region feature + segment embedding + position embedding
     - Text feature: same as BERT



##### Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training

AAAI 2020 [[paper]]((https://arxiv.org/abs/1908.06066))

   - Architecture: Single stream BERT
     <img src="images\Unicoder-VL.png" style="zoom:43%;" />

   - Pretrain dataset:

   - Pretrain tasks

     MLM + Masked Object Classification+ Visual-linguistic Matching (Cross-modality Matching) **with only [CLS] output**

   - Features
     - Image feature (Fast R-CNN)
       - [IMG] token + segment embedding + position embedding + next term
       - $(\frac{x_1}{W}, \frac{y_1}{H}, \frac{x_2}{W}, \frac{y_2}{H}, \frac{(y_2-y_1)(x_2-x_1)}{W\cdot H})$, visual feature --separately--> embedding space using FC, then added up
     - Text feature: same as BERT



##### Unified Vision-Language Pre-Training for Image Captioning and VQA

AAAI 2020, [[code]](https://github.com/LuoweiZhou/VLP), (**VLP**)



##### UNITER: Learning Universal Image-text Representations

ECCV 2020 [[paper]]((https://arxiv.org/abs/1909.11740)) [[code]](https://github.com/ChenRocks/UNITER)



##### Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks

arXiv 2020/04, ECCV 2020 [[paper]]((https://arxiv.org/pdf/2004.06165.pdf)) [[code]](https://github.com/microsoft/Oscar)





### Image-Text Retrieval & Matching

##### ImageBERT: Cross-Modal Pre-training with Large-scale Weak-supervised Image-text Data

arXiv 2020/01 [[paper]](https://arxiv.org/abs/2001.07966)

##### CROSS-PROBE BERT FOR EFFICIENT AND EFFECTIVE CROSS-MODAL SEARCH

ICLR 2021 submission. [[paper]](https://openreview.net/forum?id=bW9SYKHcZiz)

##### Multi-Modality Cross Attention Network for Image and Sentence Matching

ICCV 2020 [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Wei_Multi-Modality_Cross_Attention_Network_for_Image_and_Sentence_Matching_CVPR_2020_paper.html)



### Analysis

##### 12-in-1: Multi-Task Vision and Language Representation Learning

CVPR 2020 [[paper]]((https://arxiv.org/abs/1912.02315)) [[code]](https://github.com/facebookresearch/vilbert-multi-task) Multi-task Learning

##### Are we pretraining it right? Digging deeper into visio-linguistic pretraining

arXiv 2020/04 [[paper]](https://arxiv.org/abs/2004.08744) In-depth Analysis

##### Behind the Scene: Revealing the Secrets of Pre-trained Vision-and-Language Models 
arXiv 2020/05, ECCV 2020 Spotlight [[paper]](https://arxiv.org/abs/2005.07310) In-depth Analysis

##### Adaptive Transformers for Learning Multimodal Representations 

ACL SRW 2020 [[paper]](https://arxiv.org/abs/2005.07486) Adaptive Transformer Analysis





### Survey

##### Pre-trained Models for Natural Language Processing: A Survey 

arXiv 2020/03 [[paper]](https://arxiv.org/abs/2003.08271)

##### A Survey on Contextual Embeddings

arXiv 2020/03 [[paper]](https://arxiv.org/abs/2003.07278)

##### Trends in Integration of Vision and Language Research: A Survey of Tasks, Datasets, and Methods 

arXiv 2019 [[paper]](https://arxiv.org/abs/1907.09358)

##### Deep Multimodal Representation Learning: A Survey 

arXiv 2019 [[paper]](https://ieeexplore.ieee.org/abstract/document/8715409)



### Platforms

##### facebook MMF 

https://github.com/facebookresearch/mmf



## Transformer

### X-formers

##### Performer: "Rethinking Attention with Performers"

arXiv 2020/09 [[paper]](https://arxiv.org/abs/2009.14794)  [[code]](https://github.com/lucidrains/performer-pytorch) *Google & University of Cambreidge & DeepMind & Alan Turing Institute*

##### Linformer: "Self-Attention with Linear Complexity"

arXiv 2020/06 [[paper]](https://arxiv.org/pdf/2006.04768.pdf)  [[code]](https://github.com/tatp22/linformer-pytorch) *FAIR* 

##### Linear Transformer: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"

ICML 2020 [[paper]](https://arxiv.org/abs/2006.16236)  [[code]](https://github.com/lucidrains/linear-attention-transformer) *Idiap Research Institut*

##### Synthesizer: "Neural Speech Synthesis with Transformer Network"

arXiv 2019/02 [[paper]](https://arxiv.org/abs/1809.08895)  [[code]](https://github.com/soobinseo/Transformer-TTS) *UESTC, MSRA*

##### Sinkhorn Transformer: "Sparse Sinkhorn Attention"

arXiv 2020/02 [[paper]](https://arxiv.org/abs/2002.11296)  [[code]](https://github.com/lucidrains/sinkhorn-transformer) *Google AI*

##### Reformer: The Efficient Transformer

ICLR 2020 [[paper]](https://openreview.net/pdf?id=rkgNKkHtvB)  [[code]](https://github.com/lucidrains/reformer-pytorch) *UCB & Google Research*

##### Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context 

arXiv 2019/06 [[paper]](http://arxiv.org/abs/1901.02860) [[code]](https://github.com/kimiyoung/transformer-xl) *CMU, Google Brain*

##### Compressive Transformers for Long-Range Sequence Modelling

arXiv 2019/11 [[paper]](https://openreview.net/forum?id=SylKikSYDH)  [[code]](https://github.com/lucidrains/compressive-transformer-pytorch) *Deep Mind, UCL*

##### Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks

PMLR 2019 [[paper]](http://proceedings.mlr.press/v97/lee19d.html)  *University of Oxford*

##### Longformer: The Long-Document Transformer

arXiv 2020/04 [[paper]](https://arxiv.org/abs/2004.05150) [[code]](https://github.com/allenai/longformer) *Allen Institute for Artificial Intelligence*

##### Routing Transformer: Efficient Content-Based Sparse Attention with Routing Transformers

arXiv 2020/10 [[paper]](https://arxiv.org/pdf/2003.05997.pdf)  [[code]](https://github.com/lucidrains/routing-transformer) *Google Research*

##### Big Bird: Transformers for Longer Sequences

NIPS 2020 [[paper]](https://proceedings.neurips.cc//paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf)  *Google Research*

##### Etc: Encoding long and structured data in transformers

EMNLP 2020 [[paper]](https://arxiv.org/pdf/2004.08483.pdf)  [[code]](https://github.com/google-research/google-research/tree/master/etcmodel) Google Research

##### Memory Compressed: Generating Wikipedia by Summarizing Long Sequences

ICLR 2018 [[paper]](https://arxiv.org/abs/1801.10198) [[code]](https://github.com/lucidrains/memory-compressed-attention) ** *Google Brain*

##### Blockwise Transformer: "Blockwise Self-Attention for Long Document Understanding"

arXiv 2020/10 [[paper]](https://arxiv.org/abs/1911.02972)  [[code]](https://github.com/xptree/BlockBERT) *Tsinghua University, FAIR*

##### Image Transformer

ICML 2018 [[paper]](https://arxiv.org/abs/1802.05751)  [[code1]](https://github.com/sahajgarg/image_transformer) [[code2]](https://github.com/tensorflow/tensor2tensor/) *Google Brain, UCB, Google AI*

##### Sparse Transformer: Generating Long Sequences with Sparse Transformers

arXiv 2019/04 [[paper]](https://arxiv.org/abs/1904.10509)  [[code]](https://github.com/openai/sparse_attention) *OpenAI*

##### Axial Transformer: "Axial Attention in Multidimensional Transformers"

arXiv 2019/12 [[paper]](https://arxiv.org/abs/1912.12180) [[code]](https://github.com/lucidrains/axial-attention) *UCB, Google Brain*

##### ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

ICLR 2021 under review [[paper]](https://arxiv.org/abs/2010.11929)  [[code1]](https://github.com/google-research/vision_transformer) [[code2]](https://github.com/lucidrains/vit-pytorch)



### Survey

##### Pre-trained Models for Natural Language Processing: A Survey

arXiv 2020/03 [[paper]](https://arxiv.org/abs/2003.08271)

##### A Survey on Contextual Embeddings

arXiv 2020/03 [[paper]](https://arxiv.org/abs/2003.07278)

##### Trends in Integration of Vision and Language Research: A Survey of Tasks, Datasets, and Methods

arXiv 2019 [[paper]](https://arxiv.org/abs/1907.09358)

##### Deep Multimodal Representation Learning: A Survey

arXiv 2019 [[paper]](https://ieeexplore.ieee.org/abstract/document/8715409)

##### Multimodal Machine Learning: A Survey and Taxonomy

TPAMI 2018 [[paper]](https://arxiv.org/abs/1705.09406)

