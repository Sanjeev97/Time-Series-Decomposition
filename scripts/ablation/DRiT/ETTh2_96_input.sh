export CUDA_VISIBLE_DEVICES=3

model_name=DRiT
seq_len=96

if [ ! -d "./logs/$model_name" ]; then
    mkdir ./logs/$model_name
fi

# model_id=ETTh2_96_MOE96
model_id=ETTh2_96_FD96
pred_len=96


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id $model_id \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --e_layers_tn 2 \
  --d_model_tn 128 \
  --d_ff_tn 128 \
  --top_KFD 3 \
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log


# model_id=ETTh2_96_MOE192
model_id=ETTh2_96_FD192
pred_len=192

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id $model_id \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --e_layers_tn 2 \
  --d_model_tn 128 \
  --d_ff_tn 128 \
  --top_KFD 3 \
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log  


# model_id=ETTh2_96_MOE336
model_id=ETTh2_96_FD336
pred_len=336

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id $model_id \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --e_layers_tn 2 \
  --d_model_tn 128 \
  --d_ff_tn 128 \
  --top_KFD 3 \
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log    


# model_id=ETTh2_96_MOE720
model_id=ETTh2_96_FD720
pred_len=720

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id $model_id \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --e_layers_tn 2 \
  --d_model_tn 128 \
  --d_ff_tn 128 \
  --top_KFD 3 \
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log   
