export CUDA_VISIBLE_DEVICES=3

model_name=DRiT
seq_len=96

if [ ! -d "./logs/$model_name" ]; then
    mkdir ./logs/$model_name
fi

model_id=ETTm2_96_96
pred_len=96

python -u run_.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id $model_id \
  --model $model_name \
  --data ETTm2 \
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
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log  


model_id=ETTm2_96_192
pred_len=192

python -u run_.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id $model_id \
  --model $model_name \
  --data ETTm2 \
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
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log    


model_id=ETTm2_96_336
pred_len=336

python -u run_.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id $model_id \
  --model $model_name \
  --data ETTm2 \
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
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log    


model_id=ETTm2_96_720
pred_len=720

python -u run_.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id $model_id \
  --model $model_name \
  --data ETTm2 \
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
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log     
