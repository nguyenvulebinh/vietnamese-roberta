# Pre-trained embedding using RoBERTa architecture on Vietnamese corpus

## Overview

[RoBERTa](https://arxiv.org/abs/1907.11692) is an improved recipe for training BERT models that can match or exceed the performance of all of the post-BERT methods. The different between RoBERTa and BERT:

- Training the model longer, with bigger batches, over more data.
- Removing the next sentence prediction objective.
- Training on longer sequences.
- Dynamically changing the masking pattern applied to the training data.

Data to train this model is Vietnamese corpus crawled from many online newspapers: 50GB of text with approximate 7.7 billion words that crawl from many domains on the internet including news, law, entertainment, wikipedia and so on. Data was cleaned using [visen](https://github.com/nguyenvulebinh/visen) library and tokenize using [sentence piece](https://github.com/google/sentencepiece)

## Prepare environment

- Download the model using the following link: [cased model](https://bit.ly/vibert-cased), [uncased model](https://bit.ly/vibert-uncased) and put it in folder data-bin as the following folder structure::

```text
model-bin
├── cased
│   ├── dict.txt
│   ├── model.pt
│   └── sentencepiece.bpe.model
└── uncased
    ├── dict.txt
    ├── model.pt
    └── sentencepiece.bpe.model

```

- Install environment library
```bash
pip install -r requirements.txt
```

## Example usage

### Load RoBERTa model

```python
from fairseq.models.roberta import XLMRModel

# Using cased model
pretrained_path = './model-bin/cased/'

# Load RoBERTa model. That already include loading sentence piece model
roberta = XLMRModel.from_pretrained(pretrained_path, checkpoint_file='model.pt')
roberta.eval()  # disable dropout (or leave in train mode to finetune)
```

### Extract features from RoBERTa

```python
text_input = 'Đại học Bách Khoa Hà Nội.'
# Encode using roberta class
tokens_ids = roberta.encode(text_input)
assert tokens_ids.tolist() == [0, 451, 71, 3401, 1384, 168, 234, 5, 2]
# Extracted feature using roberta model
tokens_embed = roberta.extract_features(tokens_ids)
assert tokens_embed.shape == (1, 9, 512)
```

### Filling masks

RoBERTa can be used to fill \<mask\> tokens in the input.

```python
masked_line = 'Đại học <mask> Khoa Hà Nội'
roberta.fill_mask(masked_line, topk=5)

#('Đại học Bách Khoa Hà Nội', 0.9954977035522461, ' Bách'),
#('Đại học Y Khoa Hà Nội', 0.001166337518952787, ' Y'),
#('Đại học Đa Khoa Hà Nội', 0.0005696234875358641, ' Đa'),
#('Đại học Văn Khoa Hà Nội', 0.000467598409159109, ' Văn'),
#('Đại học Anh Khoa Hà Nội', 0.00035955727798864245, ' Anh')
```

## Model detail

This model was a custom version from RoBERTa base:

- Hidden layer: 4 (compare with 12 in RoBERTa base)
- Number of head: 4 (compare with 12 in RoBERTa base)
- Hidden size: 512 (compare with 768 in RoBERTa base)
- Number params: 35M
- Model size: 452MB
- Dict size: 50k words
- Two versions: cased (with dictionary case sensitive) and uncased (all word is lower)

```text
loading archive file ./model-bin/cased/
| dictionary: 56024 types
RobertaHubInterface(
  (model): RobertaModel(
    (decoder): RobertaEncoder(
      (sentence_encoder): TransformerSentenceEncoder(
        (embed_tokens): Embedding(56025, 512, padding_idx=1)
        (embed_positions): LearnedPositionalEmbedding(514, 512, padding_idx=1)
        (layers): ModuleList(
          (0): TransformerSentenceEncoderLayer(
            (self_attn): MultiheadAttention(
              (k_proj): Linear(in_features=512, out_features=512, bias=True)
              (v_proj): Linear(in_features=512, out_features=512, bias=True)
              (q_proj): Linear(in_features=512, out_features=512, bias=True)
              (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=512, out_features=1024, bias=True)
            (fc2): Linear(in_features=1024, out_features=512, bias=True)
            (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (1): TransformerSentenceEncoderLayer(
            (self_attn): MultiheadAttention(
              (k_proj): Linear(in_features=512, out_features=512, bias=True)
              (v_proj): Linear(in_features=512, out_features=512, bias=True)
              (q_proj): Linear(in_features=512, out_features=512, bias=True)
              (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=512, out_features=1024, bias=True)
            (fc2): Linear(in_features=1024, out_features=512, bias=True)
            (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (2): TransformerSentenceEncoderLayer(
            (self_attn): MultiheadAttention(
              (k_proj): Linear(in_features=512, out_features=512, bias=True)
              (v_proj): Linear(in_features=512, out_features=512, bias=True)
              (q_proj): Linear(in_features=512, out_features=512, bias=True)
              (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=512, out_features=1024, bias=True)
            (fc2): Linear(in_features=1024, out_features=512, bias=True)
            (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
          (3): TransformerSentenceEncoderLayer(
            (self_attn): MultiheadAttention(
              (k_proj): Linear(in_features=512, out_features=512, bias=True)
              (v_proj): Linear(in_features=512, out_features=512, bias=True)
              (q_proj): Linear(in_features=512, out_features=512, bias=True)
              (out_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (self_attn_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (fc1): Linear(in_features=512, out_features=1024, bias=True)
            (fc2): Linear(in_features=1024, out_features=512, bias=True)
            (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          )
        )
        (emb_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
      (lm_head): RobertaLMHead(
        (dense): Linear(in_features=512, out_features=512, bias=True)
        (layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
    )
    (classification_heads): ModuleDict()
  )
)
```