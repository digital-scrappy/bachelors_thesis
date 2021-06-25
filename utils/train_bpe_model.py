import sentencepiece as spm
spm.SentencePieceTrainer.train(input='../data/combined_data/train.src-trg', model_prefix='bpe_model', model_type= 'bpe', vocab_size = 88500, character_coverage = 1.0)
