export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name=DRiT
seq_len=96

if [ ! -d "./logs/$model_name" ]; then
    mkdir ./logs/$model_name
fi

# model_id=hydro_96_MOE_96
model_id=hydro_96_FD_96
pred_len=96

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/hydro/ \
  --data_path dataset_2015_2023.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --batch_size 64 \
  --e_layers_tn 3 \
  --d_model_tn 512 \
  --d_ff_tn 512 \
  --des '' \
  --itr 1 >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log


# model_id=hydro_96_MOE_192
model_id=hydro_96_FD_192
pred_len=192

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/hydro/ \
  --data_path dataset_2015_2023.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --batch_size 64 \
  --e_layers_tn 3 \
  --d_model_tn 512 \
  --d_ff_tn 512 \
  --des '' \
  --itr 1 >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log

# model_id=hydro_96_MOE_336
model_id=hydro_96_FD_336
pred_len=336

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/hydro/ \
  --data_path dataset_2015_2023.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --batch_size 64 \
  --e_layers_tn 3 \
  --d_model_tn 512 \
  --d_ff_tn 512 \
  --des '' \
  --itr 1 >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log


# model_id=hydro_96_MOE_720
model_id=hydro_96_FD_720
pred_len=720

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/hydro/ \
  --data_path dataset_2015_2023.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 4 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --batch_size 64 \
  --e_layers_tn 3 \
  --d_model_tn 512 \
  --d_ff_tn 512 \
  --des '' \
  --itr 1 >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log