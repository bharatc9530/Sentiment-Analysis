import transformers

DEVICE = "cuda"
MAX_LEN = 300
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
ACCUMULATION = 2
BERT_PATH = "bert_base_unchased/"
MODEL_PATH = "model.bin"
TRAINING_FILE = "IMDB Dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_low_case=True
)
