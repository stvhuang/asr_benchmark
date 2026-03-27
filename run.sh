# for n in 1 4 16 64; do
#     # python main.py \
#     #     --device mps \
#     #     --model_name_or_path openai/whisper-large-v3 \
#     #     --batch_size $n
#     python main.py \
#         --device mps \
#         --model_name_or_path mlx-community/whisper-large-v3-mlx \
#         --batch_size $n
# done

set -x

# for model in mlx-community/whisper-large-v3-fp16 mlx-community/whisper-large-v3-4bit mlx-community/whisper-large-v3-8bit; do
#     # ls "${HOME}/mlx-examples/whisper/mlx_models/openai/whisper-large-v3_${model}"
# 
#     python main.py \
#         --device mps \
#         --batch_size 1 \
#         --model_name_or_path ${model}
#         # --model_name_or_path "${HOME}/mlx-examples/whisper/mlx_models/openai/whisper-large-v3_${model}"
# done

for model in dtypefloat16 dtypefloat16_q_qbits8  dtypefloat16_q_qbits4; do
    python main.py \
        --device mps \
        --batch_size 1 \
        --model_name_or_path "${HOME}/mlx-examples/whisper/mlx_models/openai/whisper-large-v3_${model}"
done

for model in dtypefloat16 dtypefloat16_q_qbits8  dtypefloat16_q_qbits4; do
    python main.py \
        --device mps \
        --batch_size 1 \
        --model_name_or_path "${HOME}/mlx-examples/whisper/mlx_models/openai/whisper-large-v3_${model}"
done

for n in 1 4 16 64; do
    python main.py \
        --device mps \
        --model_name_or_path openai/whisper-large-v3 \
        --batch_size $n
    # python main.py \
    #     --device mps \
    #     --model_name_or_path mlx-community/whisper-large-v3-mlx \
    #     --batch_size $n
done

for n in 1 4 16 64; do
    python main.py \
        --device mps \
        --model_name_or_path openai/whisper-large-v3 \
        --batch_size $n
    # python main.py \
    #     --device mps \
    #     --model_name_or_path mlx-community/whisper-large-v3-mlx \
    #     --batch_size $n
done
