export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name=RMM
seq_len=96

if [ ! -d "./logs/$model_name" ]; then
    mkdir ./logs/$model_name
fi

model_id=hydro_96_MOE96
# model_id=hydro_96_FD96
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
  --d_model 256 \
  --dec_in 4 \
  --des '' \
  --itr 1 >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log


model_id=hydro_96_MOE192
# model_id=hydro_96_FD192

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
  --pred_len 192 \
  --dec_in 4 \
  --des '' \
  --itr 1 >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log

model_id=hydro_96_MOE336
# model_id=hydro_96_FD336
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
  --pred_len 336 \
  --dec_in 4 \
  --des '' \
  --itr 1 >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log


model_id=hydro_96_MOE720
# model_id=hydro_96_FD720
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
  --pred_len 720 \
  --dec_in 4 \
  --des '' \
  --itr 1 >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log