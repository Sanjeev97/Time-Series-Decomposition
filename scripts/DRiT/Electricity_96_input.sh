export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name=DRiT
seq_len=96

if [ ! -d "./logs/$model_name" ]; then
    mkdir ./logs/$model_name
fi

model_id=ECL_96_96
pred_len=96

python -u run_.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --e_layers_tn 3 \
  --d_model_tn 512 \
  --d_ff_tn 512 \
  --lradj TST \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log


model_id=ECL_96_192
pred_len=192

python -u run_.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --e_layers_tn 3 \
  --d_model_tn 512 \
  --d_ff_tn 512 \
  --batch_size 16 \
  --lradj TST \
  --learning_rate 0.0005 \
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log


model_id=ECL_96_336
pred_len=336

python -u run_.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --e_layers_tn 3 \
  --d_model_tn 512 \
  --d_ff_tn 512 \
  --batch_size 16 \
  --lradj TST \
  --learning_rate 0.0005 \
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log  


model_id=ECL_96_720
pred_len=720

python -u run_.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id $model_id \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --e_layers_tn 3 \
  --d_model_tn 512 \
  --d_ff_tn 512 \
  --batch_size 16 \
  --lradj TST \
  --learning_rate 0.0005 \
  --des '' >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log    