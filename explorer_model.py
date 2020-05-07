from fairseq.models.roberta import XLMRModel
from pprint import pprint
import torch
import os
import sentencepiece as spm

pretrained_path = './model-bin/cased/'

# load sentence piece model for checking purpose
sp = spm.SentencePieceProcessor()
sp.Load(os.path.join(pretrained_path, 'sentencepiece.bpe.model'))

# Load RoBERTa model. That already include loading sentence piece model
roberta = XLMRModel.from_pretrained(pretrained_path, checkpoint_file='model.pt')
roberta.eval()  # disable dropout (or leave in train mode to finetune)
print(roberta)

text_input = 'Đại học Bách Khoa Hà Nội.'

# Encode using roberta class
tokens_ids = roberta.encode(text_input)
assert tokens_ids.tolist() == [0, 451, 71, 3401, 1384, 168, 234, 5, 2]
# Tokenizer using sentence piece
tokens_text = sp.encode_as_pieces(text_input)
assert tokens_text == ['▁Đại', '▁học', '▁Bách', '▁Khoa', '▁Hà', '▁Nội', '.']
assert roberta.decode(tokens_ids) == text_input

print(tokens_ids)
print(tokens_text)
print(roberta.decode(tokens_ids))
# Extracted feature using roberta model
tokens_embed = roberta.extract_features(tokens_ids)
assert tokens_embed.shape == (1, 9, 512)

# Filling marks
masked_line = 'Đại học <mask> Khoa Hà Nội'
topk_filled_outputs = roberta.fill_mask(masked_line, topk=5)
pprint(topk_filled_outputs)
