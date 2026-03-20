from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC, Lowercase, Sequence
import json
import os

with open('config.json') as f:
    cf = json.load(f)

INPUT_FILES = ["wiki.train.txt"]
OUTPUT_DIR = "trained_tokenizer"

os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = Tokenizer(BPE(unk_token="<unk>"))

tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=cf["size"],
    min_frequency=1,
    special_tokens=cf["special_tokens"],
)

print("Training tokenizer")
tokenizer.train(INPUT_FILES, trainer)
tokenizer.save(os.path.join(OUTPUT_DIR, "tokenizer.json"))
