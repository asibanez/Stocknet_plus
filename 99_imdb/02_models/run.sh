#INPUT_DIR=C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/06_stocknet/00_1_data_imbd/01_preprocessed
#OUTPUT_DIR=C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/06_stocknet/00_1_data_imbd/02_runs/00_TEST_0

INPUT_DIR=/data/users/sibanez/05_IMBD/02_preprocessed_att_mask
OUTPUT_DIR=/data/users/sibanez/05_IMBD/02_runs/01_TEST_1_att_masks

MODEL_FILENAME=model_v1.py

python train_test.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --model_filename=$MODEL_FILENAME \
    --task=Test \
    \
    --model_name=bert-base-uncased \
    --seq_len=512 \
    --hidden_dim=768 \
    --freeze_BERT=False \
    --seed=1234 \
    --use_cuda=True \
    \
    --n_epochs=10 \
    --batch_size_train=60 \
    --shuffle_train=True \
    --drop_last_train=True \
    --dev_train_ratio=1 \
    --train_toy_data=False \
    --len_train_toy_data=100 \
    --lr=2e-6 \
    --wd=1e-6 \
    --dropout=0.2 \
    --momentum=0.9 \
    --save_final_model=True \
    --save_model_steps=True \
    --save_step_cliff=0 \
    --gpu_ids_train=0,1 \
    \
    --test_file=model_test.pkl \
    --model_file=model.pt.4 \
    --batch_size_test=30 \
    --gpu_id_test=1 \

#read -p 'EOF'

#--model_name=nlpaueb/legal-bert-small-uncased \
#--hidden_dim=512 \

#--task=Train / Test
#--pooing=Avg / Max
#--batch_size=280 / 0,1,2,3
#--wd=1e-6
