export CUDA_VISIBLE_DEVICES=1

model_name=DRTimesNet
seq_len=96

if [ ! -d "./logs/$model_name" ]; then
    mkdir ./logs/$model_name
fi

model_id=weather_96_96
pred_len=96

#enc,dec,c_out
python -u run_.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --d_layers 3 \
  --dec_in 21 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --batch_size 64 \
  --e_layers_tn 2 \
  --d_layers_tn 1 \
  --factor_tn 3 \
  --d_model_tn 32 \
  --d_ff_tn 32 \
  --top_k_tn 5 \
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log


model_id=weather_96_192
pred_len=192

python -u run_.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id $model_id\
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --d_layers 3 \
  --dec_in 21 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --batch_size 64 \
  --e_layers_tn 2 \
  --d_layers_tn 1 \
  --factor_tn 3 \
  --d_model_tn 32 \
  --d_ff_tn 32 \
  --top_k_tn 5 \
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log


model_id=weather_96_336
pred_len=336

python -u run_.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --d_layers 3 \
  --dec_in 21 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --batch_size 64 \
  --e_layers_tn 2 \
  --d_layers_tn 1 \
  --factor_tn 3 \
  --d_model_tn 32 \
  --d_ff_tn 32 \
  --top_k_tn 5 \
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log


model_id=weather_96_720
pred_len=720

python -u run_.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --d_layers 3 \
  --dec_in 21 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --batch_size 64 \
  --e_layers_tn 2 \
  --d_layers_tn 1 \
  --factor_tn 3 \
  --d_model_tn 32 \
  --d_ff_tn 32 \
  --top_k_tn 5 \
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log
