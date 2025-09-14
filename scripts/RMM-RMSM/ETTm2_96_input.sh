export CUDA_VISIBLE_DEVICES=3

model_name=RMSM
seq_len=96

if [ ! -d "./logs/$model_name" ]; then
    mkdir ./logs/$model_name
fi

model_id=ETTm2_96_96
pred_len=96

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id $model_id \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --dec_in 7 \
  --itr 1 \
  --d_model 256 \
  --batch_size 128 \
  --topK 5 \
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log  


model_id=ETTm2_96_192
pred_len=192

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id $model_id \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --dec_in 7 \
  --itr 1 \
  --d_model 256 \
  --batch_size 128 \
  --topK 5 \
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log    


model_id=ETTm2_96_336
pred_len=336

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id $model_id \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --dec_in 7 \
  --itr 1 \
  --d_model 256 \
  --batch_size 128 \
  --topK 5 \
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log    


model_id=ETTm2_96_720
pred_len=720

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id $model_id \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --dec_in 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 256 \
  --batch_size 128 \
  --topK 5 \
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log     
