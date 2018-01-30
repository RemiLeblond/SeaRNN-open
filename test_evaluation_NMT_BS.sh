#!/usr/bin/env bash
# MLE, dropout, dev, BS 1 (best checkpoints on valid: 36.6, 39.2)
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/MLE_dropout/checkpoint_iter_100000.pth
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/MLE_dropout/checkpoint_iter_100000.pth --beam_size 1 --decoding beam
# MLE, dropout, test, BS 1
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/MLE_dropout/checkpoint_iter_100000.pth
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/MLE_dropout/checkpoint_iter_100000.pth --beam_size 1 --decoding beam
# MLE, dropout, dev, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/MLE_dropout/checkpoint_iter_100000.pth --beam_size 10 --decoding beam
# MLE, dropout, test, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/MLE_dropout/checkpoint_iter_100000.pth --beam_size 10 --decoding beam

# TL dropout, dev, BS 1 (best checkpoints 98.1, 99.2)
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/TL_dropout/checkpoint_iter_100000.pth
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/TL_dropout/checkpoint_iter_100000.pth --beam_size 1 --decoding beam
# TL dropout, test, BS 1
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/TL_dropout/checkpoint_iter_100000.pth
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/TL_dropout/checkpoint_iter_100000.pth --beam_size 1 --decoding beam
# TL dropout, dev, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/TL_dropout/checkpoint_iter_100000.pth --beam_size 10 --decoding beam --beam_scaling 2.4
# TL dropout, test, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/TL_dropout/checkpoint_iter_100000.pth --beam_size 10 --decoding beam --beam_scaling 2.4

# Mixed, KL_100, dev, BS 1
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_kl_100_0.3/checkpoint_iter_81600.pth --beam_size 1 --decoding beam
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_kl_100_0.3/checkpoint_iter_95300.pth --beam_size 1 --decoding beam
# Mixed, KL_100, test, BS 1
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_kl_100_0.3/checkpoint_iter_81600.pth --beam_size 1 --decoding beam
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_kl_100_0.3/checkpoint_iter_95300.pth --beam_size 1 --decoding beam
# Mixed, KL_100, dev, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_kl_100_0.3/checkpoint_iter_81600.pth --beam_size 10 --decoding beam --beam_scaling 2.4
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_kl_100_0.3/checkpoint_iter_95300.pth --beam_size 10 --decoding beam --beam_scaling 2.4
# Mixed, KL_100, test, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_kl_100_0.3/checkpoint_iter_81600.pth --beam_size 10 --decoding beam --beam_scaling 2.4
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_kl_100_0.3/checkpoint_iter_95300.pth --beam_size 10 --decoding beam --beam_scaling 2.4

# Mixed, KL_200, dev, BS 1
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_kl_200_0.3/checkpoint_iter_96200.pth --beam_size 1 --decoding beam
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_kl_200_0.3/checkpoint_iter_80900.pth --beam_size 1 --decoding beam
# Mixed, KL_200, test, BS 1
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_kl_200_0.3/checkpoint_iter_96200.pth --beam_size 1 --decoding beam
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_kl_200_0.3/checkpoint_iter_80900.pth --beam_size 1 --decoding beam
# Mixed, KL_200, dev, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_kl_200_0.3/checkpoint_iter_96200.pth --beam_size 10 --decoding beam --beam_scaling 2.4
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_kl_200_0.3/checkpoint_iter_80900.pth --beam_size 10 --decoding beam --beam_scaling 2.4
# Mixed, KL_200, test, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_kl_200_0.3/checkpoint_iter_96200.pth --beam_size 10 --decoding beam --beam_scaling 2.4
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_kl_200_0.3/checkpoint_iter_80900.pth --beam_size 10 --decoding beam --beam_scaling 2.4

# Mixed, TL, dev, BS 1
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_target-learning_100_0.3/checkpoint_iter_76400.pth --beam_size 1 --decoding beam
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_target-learning_100_0.3/checkpoint_iter_99300.pth --beam_size 1 --decoding beam
# Mixed, TL, test, BS 1
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_target-learning_100_0.3/checkpoint_iter_76400.pth --beam_size 1 --decoding beam
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_target-learning_100_0.3/checkpoint_iter_99300.pth --beam_size 1 --decoding beam
# Mixed, TL, dev, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_target-learning_100_0.3/checkpoint_iter_76400.pth --beam_size 10 --decoding beam --beam_scaling 2.4
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_target-learning_100_0.3/checkpoint_iter_99300.pth --beam_size 10 --decoding beam --beam_scaling 2.4
# Mixed, TL, test, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_target-learning_100_0.3/checkpoint_iter_76400.pth --beam_size 10 --decoding beam --beam_scaling 2.4
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/mixed_target-learning_100_0.3/checkpoint_iter_99300.pth --beam_size 10 --decoding beam --beam_scaling 2.4

# MLE, no nmt/dropout, dev, BS 1
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/MLE_nodropout/checkpoint_iter_9600.pth --beam_size 1 --decoding beam
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/MLE_nodropout/checkpoint_iter_16300.pth --beam_size 1 --decoding beam
# MLE, no dropout, test, BS 1
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/MLE_nodropout/checkpoint_iter_9600.pth --beam_size 1 --decoding beam
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/MLE_nodropout/checkpoint_iter_16300.pth --beam_size 1 --decoding beam
# MLE, no dropout, dev, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/MLE_nodropout/checkpoint_iter_9600.pth --beam_size 10 --decoding beam
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/MLE_nodropout/checkpoint_iter_16300.pth --beam_size 10 --decoding beam
# MLE, no dropout, test, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/MLE_nodropout/checkpoint_iter_9600.pth --beam_size 10 --decoding beam
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/MLE_nodropout/checkpoint_iter_16300.pth --beam_size 10 --decoding beam

# TL, no dropout, dev, BS 1
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/TL_nodropout/checkpoint_iter_21400.pth --beam_size 1 --decoding beam
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/TL_nodropout/checkpoint_iter_21600.pth --beam_size 1 --decoding beam
# TL, no dropout, test, BS 1
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/TL_nodropout/checkpoint_iter_21400.pth --beam_size 1 --decoding beam
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/TL_nodropout/checkpoint_iter_21600.pth --beam_size 1 --decoding beam
# TL, no dropout, dev, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/TL_nodropout/checkpoint_iter_21400.pth --beam_size 10 --decoding beam --beam_scaling 2.4
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/TL_nodropout/checkpoint_iter_21600.pth --beam_size 10 --decoding beam --beam_scaling 2.4
# TL, no dropout, test, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/TL_nodropout/checkpoint_iter_21400.pth --beam_size 10 --decoding beam --beam_scaling 2.4
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/TL_nodropout/checkpoint_iter_21600.pth --beam_size 10 --decoding beam --beam_scaling 2.4

# KL 200, no dropout, dev, BS 1
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/mixed_kl_200_0.0/checkpoint_iter_19200.pth --beam_size 1 --decoding beam
# KL 200, no dropout, test, BS 1
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/mixed_kl_200_0.0/checkpoint_iter_19200.pth --beam_size 1 --decoding beam
# KL 200, no dropout, dev, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/mixed_kl_200_0.0/checkpoint_iter_19200.pth --beam_size 10 --decoding beam --beam_scaling 2.4
# KL 200, no dropout, test, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/mixed_kl_200_0.0/checkpoint_iter_19200.pth --beam_size 10 --decoding beam --beam_scaling 2.4

# KL 100, no dropout, dev, BS 1
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/mixed_kl_100_0.0/checkpoint_iter_19100.pth --beam_size 1 --decoding beam
# KL 100, no dropout, test, BS 1
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/mixed_kl_100_0.0/checkpoint_iter_19100.pth --beam_size 1 --decoding beam
# KL 100, no dropout, dev, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/mixed_kl_100_0.0/checkpoint_iter_19100.pth --beam_size 10 --decoding beam --beam_scaling 2.4
# KL 100, no dropout, test, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/mixed_kl_100_0.0/checkpoint_iter_19100.pth --beam_size 10 --decoding beam --beam_scaling 2.4

# TL, no dropout, dev, BS 1
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/mixed_target-learning_100_0.0/checkpoint_iter_18900.pth --beam_size 1 --decoding beam
# TL, no dropout, test, BS 1
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/mixed_target-learning_100_0.0/checkpoint_iter_18900.pth --beam_size 1 --decoding beam
# TL, no dropout, dev, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/mixed_target-learning_100_0.0/checkpoint_iter_18900.pth --beam_size 10 --decoding beam --beam_scaling 2.4
# TL, no dropout, test, BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_test.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/no_dropout/mixed_target-learning_100_0.0/checkpoint_iter_18900.pthh --beam_size 10 --decoding beam --beam_scaling 2.4





# MLE, dropout, dev, BS 1 (best checkpoints on valid: 36.6, 39.2)
Hamming = 0.7962, Seq = 0.9987, Log loss = 469.6249, Train loss (hamming): 0.7963, BLEU = 0.4488, Eval time: 4.6s
Hamming = 0.7937, Seq = 0.9663, Log loss = 239.1812, Train loss (hamming): 0.7364, BLEU = 0.2879, Eval time: 13.9s
# MLE, dropout, test, BS 1
Hamming = 0.7962, Seq = 0.9987, Log loss = 469.6249, Train loss (hamming): 0.7963, BLEU = 0.4488, Eval time: 4.3s
Hamming = 0.7953, Seq = 0.9593, Log loss = 229.1648, Train loss (hamming): 0.7405, BLEU = 0.2745, Eval time: 12.8s
# MLE, dropout, dev, BS 10
Hamming = 0.7834, Seq = 0.9953, Log loss = 469.6249, Train loss (hamming): 0.7835, BLEU = 0.4734, Eval time: 120.6s
Hamming = 0.7679, Seq = 0.9643, Log loss = 239.1812, Train loss (hamming): 0.7084, BLEU = 0.2987, Eval time: 275.9s
# MLE, dropout, test, BS 10
Hamming = 0.7834, Seq = 0.9953, Log loss = 469.6249, Train loss (hamming): 0.7835, BLEU = 0.4734, Eval time: 124.2s
Hamming = 0.7681, Seq = 0.9574, Log loss = 229.1648, Train loss (hamming): 0.7111, BLEU = 0.2861, Eval time: 279.8s

# TL dropout, dev, BS 1 (best checkpoints 98.1, 99.2)
Hamming = 0.8572, Seq = 1.0000, Log loss = 444.6449, Train loss (hamming): 0.8572, BLEU = 0.3137, Eval time: 4.3s
Hamming = 0.7870, Seq = 0.9627, Log loss = 383.4967, Train loss (hamming): 0.7285, BLEU = 0.2951, Eval time: 14.1s
# TL dropout, test, BS 1
Hamming = 0.8572, Seq = 1.0000, Log loss = 444.6449, Train loss (hamming): 0.8572, BLEU = 0.3137, Eval time: 4.2s
Hamming = 0.7861, Seq = 0.9602, Log loss = 384.9650, Train loss (hamming): 0.7313, BLEU = 0.2797, Eval time: 13.0s
# TL dropout, dev, BS 10
Hamming = 0.8598, Seq = 1.0000, Log loss = 444.6449, Train loss (hamming): 0.8597, BLEU = 0.2911, Eval time: 120.4s
Hamming = 0.7897, Seq = 0.9669, Log loss = 383.4967, Train loss (hamming): 0.7313, BLEU = 0.2759, Eval time: 264.7s
# TL dropout, test, BS 10
Hamming = 0.8598, Seq = 1.0000, Log loss = 444.6449, Train loss (hamming): 0.8597, BLEU = 0.2911, Eval time: 122.7s
Hamming = 0.7889, Seq = 0.9627, Log loss = 384.9650, Train loss (hamming): 0.7351, BLEU = 0.2619, Eval time: 265.9s

# Mixed, KL_100, dev, BS 1
Hamming = 0.8643, Seq = 1.0000, Log loss = 445.2295, Train loss (hamming): 0.8643, BLEU = 0.3004, Eval time: 33.8s
Hamming = 0.7900, Seq = 0.9628, Log loss = 332.9733, Train loss (hamming): 0.7315, BLEU = 0.2935, Eval time: 84.4s
Hamming = 0.8625, Seq = 1.0000, Log loss = 443.6750, Train loss (hamming): 0.8624, BLEU = 0.3004, Eval time: 33.7s
Hamming = 0.7907, Seq = 0.9631, Log loss = 332.9884, Train loss (hamming): 0.7322, BLEU = 0.2946, Eval time: 84.5s
# Mixed, KL_100, test, BS 1
Hamming = 0.8643, Seq = 1.0000, Log loss = 445.2295, Train loss (hamming): 0.8643, BLEU = 0.3004, Eval time: 34.6s
Hamming = 0.7908, Seq = 0.9590, Log loss = 330.7145, Train loss (hamming): 0.7370, BLEU = 0.2790, Eval time: 85.0s
Hamming = 0.8625, Seq = 1.0000, Log loss = 443.6750, Train loss (hamming): 0.8624, BLEU = 0.3004, Eval time: 33.2s
Hamming = 0.7908, Seq = 0.9593, Log loss = 330.9183, Train loss (hamming): 0.7376, BLEU = 0.2784, Eval time: 82.3s
# Mixed, KL_100, dev, BS 10
Hamming = 0.8667, Seq = 1.0000, Log loss = 445.2295, Train loss (hamming): 0.8667, BLEU = 0.2868, Eval time: 124.5s
Hamming = 0.7879, Seq = 0.9651, Log loss = 332.9733, Train loss (hamming): 0.7296, BLEU = 0.2780, Eval time: 289.8s
Hamming = 0.8672, Seq = 1.0000, Log loss = 443.6750, Train loss (hamming): 0.8671, BLEU = 0.2857, Eval time: 123.3s
Hamming = 0.7900, Seq = 0.9661, Log loss = 332.9884, Train loss (hamming): 0.7314, BLEU = 0.2799, Eval time: 286.1s
# Mixed, KL_100, test, BS 10
Hamming = 0.8667, Seq = 1.0000, Log loss = 445.2295, Train loss (hamming): 0.8667, BLEU = 0.2868, Eval time: 125.4s
Hamming = 0.7897, Seq = 0.9630, Log loss = 330.7145, Train loss (hamming): 0.7359, BLEU = 0.2662, Eval time: 287.7s

# Mixed, KL_200, dev, BS 1


# Mixed, KL_200, test, BS 1


# Mixed, KL_200, dev, BS 10


# Mixed, KL_200, test, BS 10



# Mixed, TL, dev, BS 1

# Mixed, TL, test, BS 1

# Mixed, TL, dev, BS 10

# Mixed, TL, test, BS 10


# MLE, no nmt/dropout, dev, BS 1

# MLE, no dropout, test, BS 1

# MLE, no dropout, dev, BS 10

# MLE, no dropout, test, BS 10


# TL, no dropout, dev, BS 1

# TL, no dropout, test, BS 1

# TL, no dropout, dev, BS 10

# TL, no dropout, test, BS 10


# BS 1 for greedy results
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/TL_dropout/checkpoint_iter_100000.pth --beam_size 1 --decoding beam
# BS 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/TL_dropout/checkpoint_iter_100000.pth --decoding beam --beam_size 10 --beam_scaling 1
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/TL_dropout/checkpoint_iter_100000.pth --decoding beam --beam_size 10 --beam_scaling 2
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/TL_dropout/checkpoint_iter_100000.pth --decoding beam --beam_size 10 --beam_scaling 4
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/TL_dropout/checkpoint_iter_100000.pth --decoding beam --beam_size 10 --beam_scaling 6
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/TL_dropout/checkpoint_iter_100000.pth --decoding beam --beam_size 10 --beam_scaling 8
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/TL_dropout/checkpoint_iter_100000.pth --decoding beam --beam_size 10 --beam_scaling 10
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/TL_dropout/checkpoint_iter_100000.pth --decoding beam --beam_size 10 --beam_scaling 20
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/TL_dropout/checkpoint_iter_100000.pth --decoding beam --beam_size 10 --beam_scaling 40
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/TL_dropout/checkpoint_iter_100000.pth --decoding beam --beam_size 10 --beam_scaling 70
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/TL_dropout/checkpoint_iter_100000.pth --decoding beam --beam_size 10 --beam_scaling 100
python main-seq2seq.py --dataset nmt --dataroot /sequoia/data2/rleblond/data/iwslt14/iwlst14_de-en_train_dev.train.pt --revert_input_sequence --memory_size 256 --memory_size_encoder 256 --rnn_depth 1 --bidirectional --attention --attn_type sum-tanh --max_iter 0 --print_iter 1 --eval_iter 100 --save_iter 100 --eval_size 1500 --checkpoint_file /sequoia/data2/rleblond/official_results/nmt/dropout/TL_dropout/checkpoint_iter_100000.pth --decoding beam --beam_size 10 --beam_scaling 1000