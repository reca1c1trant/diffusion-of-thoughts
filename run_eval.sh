# export CUDA_VISIBLE_DEVICES=0,1,2,3
# results are slightly diff with that in paper due to code reformulation

# dot (T=64 by default)
python3 evaluation_batch.py --weights_path outputs/gsm8k-bs128-fix_src-digit-steps120000 --fix_src --digit --dataset gsm8k --score_temp 0.5

# dot dpmsolver 
python3 evaluation_batch.py --weights_path outputs/gsm8k-bs128-fix_src-digit-steps120000 --fix_src --digit --dataset gsm8k --dpm_solver

# dot T=8
python3 evaluation_batch.py --weights_path outputs/gsm8k-bs128-fix_src-digit-steps120000 --fix_src --digit --dataset gsm8k --score_temp 0.5 --sampling_timesteps 8

# dot dpmsolver T=8
python3 evaluation_batch.py --weights_path outputs/gsm8k-bs128-fix_src-digit-steps120000 --fix_src --digit --dataset gsm8k --dpm_solver --sampling_timesteps 8

# dot self-consistency
python3 evaluation_batch.py --weights_path outputs/gsm8k-bs128-fix_src-digit-steps120000 --fix_src --digit --dataset gsm8k --score_temp 0.8 --runs 20 --apply_sc --logit_sample --logit_temp 0.5

# mp-dot
python3 evaluation_batch.py --weights_path outputs/gsm8k-bs128-fix_src-cot-digit-steps31000 --fix_src --digit --cot --dataset gsm8k --score_temp 0.5

# mp-dot dpm solver
python3 evaluation_batch.py --weights_path outputs/gsm8k-bs128-fix_src-cot-digit-steps31000 --fix_src --digit --cot --dataset gsm8k --dpm_solver 

# mp-dot dpm solver T=4
python3 evaluation_batch.py --weights_path outputs/gsm8k-bs128-fix_src-cot-digit-steps31000 --fix_src --digit --cot --dataset gsm8k --dpm_solver --sampling_timesteps 4

# mp-dot self-consistency
python3 evaluation_batch.py --weights_path outputs/gsm8k-bs128-fix_src-cot-digit-steps31000 --fix_src --digit --cot --dataset gsm8k --score_temp 0.8 --runs 20 --apply_sc --logit_sample --logit_temp 0.5 


# 使用SelfCheck的评估
python evaluation_batch.py --weights_path outputs/gsm8k-bs128-fix_src-cot-digit-steps31000 --fix_src --digit --cot --dataset gsm8k --score_temp 0.5 --use_selfcheck --selfcheck_threshold 0.7 --selfcheck_apply_corrections

# 使用更高的置信度阈值
python evaluation_batch.py --weights_path outputs/gsm8k-bs128-fix_src-cot-digit-steps31000 --fix_src --digit --cot --dataset gsm8k --score_temp 0.5 --use_selfcheck --selfcheck_threshold 0.8 --selfcheck_apply_corrections

# 仅验证不修正
python evaluation_batch.py --weights_path outputs/gsm8k-bs128-fix_src-cot-digit-steps31000 --fix_src --digit --cot --dataset gsm8k --score_temp 0.5 --use_selfcheck --selfcheck_threshold 0.7 --selfcheck_apply_corrections False

# 与自洽推理(self-consistency)结合使用
python evaluation_batch.py --weights_path outputs/gsm8k-bs128-fix_src-cot-digit-steps31000 --fix_src --digit --cot --dataset gsm8k --score_temp 0.5 --use_selfcheck --selfcheck_threshold 0.7 --runs 5 --apply_sc