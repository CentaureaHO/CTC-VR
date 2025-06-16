# High prec
python rnnt_train.py --streaming --static_chunk_size 32 --use_dynamic_chunk \
    --num_decoding_left_chunks 6 --ctc_weight 0.3 --batch_size 12 --epochs 100

# middle prec
# python rnnt_train.py --streaming --static_chunk_size 16 --use_dynamic_chunk \
#     --num_decoding_left_chunks 3 --ctc_weight 0.4 --batch_size 16 --epochs 100

# fast
# python rnnt_train.py --streaming --static_chunk_size 20 --use_dynamic_chunk \
#     --num_decoding_left_chunks 4 --ctc_weight 0.3 --batch_size 16 --epochs 100