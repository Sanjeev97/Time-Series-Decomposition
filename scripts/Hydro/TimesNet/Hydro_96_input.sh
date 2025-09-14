export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name=TimesNet
seq_len=96

if [ ! -d "./logs/$model_name" ]; then
    mkdir ./logs/$model_name
fi

model_id=hydro_96_96
pred_len=96

python -u runTimesNet.py \
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
  --d_ff 512 \
  --dec_in 4 \
  --n_heads 32 \
  --QAM_end 0.2 \
  --batch_size 64 \
  --e_layers_tn 2 \
  --d_layers_tn 1 \
  --factor_tn 3 \
  --d_model_tn 32 \
  --d_ff_tn 32 \
  --des 'decay_loss' \
  --itr 1 >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log


model_id=hydro_96_192
pred_len=192

python -u runTimesNet.py \
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
  --d_layers 1 \
  --factor 3 \
  --dec_in 4 \
  --des 'decay_loss' \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --QAM_end 0.3 \
  --batch_size 64 \
  --e_layers_tn 2 \
  --d_layers_tn 1 \
  --factor_tn 3 \
  --d_model_tn 32 \
  --d_ff_tn 32 \
  --itr 1 >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log

model_id=hydro_96_336
pred_len=336

python -u runTimesNet.py \
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
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --dec_in 4 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --QAM_end 0.5 \
  --batch_size 64 \
  --e_layers_tn 2 \
  --d_layers_tn 1 \
  --factor_tn 3 \
  --d_model_tn 32 \
  --d_ff_tn 32 \
  --des 'decay_loss' \
  --itr 1 >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log


model_id=hydro_96_720
pred_len=720

python -u runTimesNet.py \
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
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --dec_in 4 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --QAM_end 0.7 \
  --batch_size 64 \
  --e_layers_tn 2 \
  --d_layers_tn 1 \
  --factor_tn 3 \
  --d_model_tn 32 \
  --d_ff_tn 32 \
  --des 'decay_loss' \
  --itr 1 >./logs/$model_name/$model_id'_'$model_name'_sl'$input_len'_pl'$pred_len.log