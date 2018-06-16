
DATA_ROOT=data/iwlst14_de-en

python preprocess_nmt.py -train_src ${DATA_ROOT}/train.de-en.de -train_tgt ${DATA_ROOT}/train.de-en.en -valid_src ${DATA_ROOT}/dev.de-en.de -valid_tgt ${DATA_ROOT}/dev.de-en.en -save_data ${DATA_ROOT}/iwlst14_de-en_train_dev -lower -shuffle 0 -sort_data 0 -seq_length 50 -src_vocab_size 32009 -tgt_vocab_size 22822

python preprocess_nmt.py -train_src ${DATA_ROOT}/train.de-en.de -train_tgt ${DATA_ROOT}/train.de-en.en -valid_src ${DATA_ROOT}/test.de-en.de -valid_tgt ${DATA_ROOT}/test.de-en.en -save_data ${DATA_ROOT}/iwlst14_de-en_train_test -lower -shuffle 0 -sort_data 0 -seq_length 50 -src_vocab ${DATA_ROOT}/iwlst14_de-en_train_dev.src.dict -tgt_vocab ${DATA_ROOT}/iwlst14_de-en_train_dev.tgt.dict
