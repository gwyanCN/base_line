val_path_="/media/b227/ygw/Dcase2023/data/Development_Set_21/Validation_Set"
ref_files_path_="/media/b227/ygw/Dcase2023/data/Development_Set_21/Validation_Set" 

if [[ $1 == 0 ]] || [[ $1 == 1 ]];then
    python utils/pd_concat.py
    python utils/post_proc.py -val_path $val_path_ -evaluation_file output_csv/tim/Eval_out_tim.csv -new_evaluation_file  output_csv/tim/Eval_out_tim_post.csv
    cd ../evaluation_metrics
    python -m evaluation -pred_file=../src/output_csv/tim/Eval_out_tim_post.csv -ref_files_path=$ref_files_path_ -team_name=PKU_ADSP -dataset=VAL -savepath=../evaluation_metrics/
    cd ../src
elif [[ $1==2 ]] || [[ $1==3 ]];then
    python utils/post_proc.py -val_path=$val_path_ -evaluation_file output_csv/ori/Eval_out_ori.csv -new_evaluation_file output_csv/ori/Eval_out_ori_post.csv
    cd ../evaluation_metrics
    python -m evaluation -pred_file=../src/output_csv/ori/Eval_out_ori_post.csv -ref_files_path=$ref_files_path_ -team_name=PKU_ADSP -dataset=VAL -savepath=../evaluation_metrics/
    cd ../src
fi