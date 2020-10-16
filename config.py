import transformers

DEVICE = "cpu"
MAX_LEN = 32
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
ACCUMULATION = 2
BERT_PATH = "input/bert_base_unchased/"
MODEL_PATH = "pytorch_model.bin"
TRAINING_FILE = "input/IMDB Dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_low_case=True
)
