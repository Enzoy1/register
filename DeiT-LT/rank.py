import pathlib
import torch
import numpy as np
import os
import matplotlib.pyplot as plt



temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def get_dist_token_rank(model_path):
    # Load the checkpoint
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get the model state dict
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Initialize lists to store results
    block_nums = []
    ranks = []
    
    # First, get the initial dist_token rank
    if 'dist_token' in state_dict:
        dist_token = state_dict['dist_token']
        print(f"\nInitial dist_token shape: {dist_token.shape}")
        # Reshape if needed and calculate rank
        dist_token_2d = dist_token.reshape(dist_token.size(0), -1)
        U, S, V = torch.svd(dist_token_2d)
        rank = torch.sum(S > 1e-5).item()
        block_nums.append(0)  # 0 represents the initial dist_token
        ranks.append(rank)
        print(f"Initial dist_token rank: {rank}")
    
    # For each block, analyze the attention output for dist_token
    num_blocks = len([k for k in state_dict.keys() if k.startswith('blocks.')])
    print(f"\nAnalyzing {num_blocks} blocks...")
    
    for i in range(num_blocks):
        qkv_key = f'blocks.{i}.attn.qkv.weight'
        proj_key = f'blocks.{i}.attn.proj.weight'
        
        if qkv_key in state_dict and proj_key in state_dict:
            # Get QKV and projection weights
            qkv_weight = state_dict[qkv_key]
            proj_weight = state_dict[proj_key]
            
            # For DeiT, the QKV weight shape is (3*dim, dim)
            dim = proj_weight.shape[0]
            
            # Analyze the attention output for dist_token
            # We focus on the projection matrix as it represents the output transformation
            U, S, V = torch.svd(proj_weight)
            rank = torch.sum(S > 1e-5).item()
            
            block_nums.append(i + 1)  # +1 to account for initial dist_token
            ranks.append(rank)
            print(f"Block {i} attention projection rank: {rank}")
    
    if not ranks:
        raise ValueError("No dist_token or attention layers found in the model.")
    
    return block_nums, ranks

def plot_ranks(block_nums, ranks):
    plt.figure(figsize=(12, 8))
    plt.plot(block_nums, ranks, 'bo-', linewidth=2, markersize=8)
    plt.grid(True)
    plt.xlabel('Block Number (0 = Initial dist_token)')
    plt.ylabel('Feature Rank')
    plt.title('Distribution of Dist Token Feature Ranks Across Blocks')
    
    # Ensure integer ticks for blocks
    plt.xticks(block_nums)
    
    # Add value labels on top of each point
    for i, rank in enumerate(ranks):
        plt.text(block_nums[i], ranks[i], str(rank), 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('dist_token_ranks.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'dist_token_ranks.png'")
    plt.close()

def main():
    model_path = input("Please enter the path to your .pth model file: ")
    
    try:
        print("\nAnalyzing model...")
        block_nums, ranks = get_dist_token_rank(model_path)
        
        print("\nRank results:")
        for block, rank in zip(block_nums, ranks):
            print(f"Block {block}: Rank = {rank}")
        
        print("\nGenerating plot...")
        plot_ranks(block_nums, ranks)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
# 提取全部参数的特征秩
# def extract_model_ranks(model_path):
#     """
#     提取 PyTorch 模型中所有参数的秩
#     :param model_path: 模型文件路径 (.pth 格式)
#     """
#     # 确保路径是字符串格式
#     model_path = str(model_path)

#     # 加载模型文件
#     checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

#     # 检查是否包含模型参数
#     if 'model' in checkpoint:
#         model_state_dict = checkpoint['model']
#     else:
#         raise KeyError("Model parameters not found in the checkpoint file.")

#     # 遍历模型的所有参数
#     for name, param in model_state_dict.items():
#         if isinstance(param, torch.Tensor) and param.dim() > 1:  # 只处理二维或更高维度的张量
#             # 将参数转换为 NumPy 数组
#             param_matrix = param.numpy()
#             # 计算秩
#             rank = np.linalg.matrix_rank(param_matrix)
#             print(f"Layer: {name}, Shape: {param_matrix.shape}, Rank: {rank}")
#         else:
#             print(f"Layer: {name}, Shape: {param.shape if isinstance(param, torch.Tensor) else 'N/A'}, Rank: N/A (not a matrix or not a tensor)")

# # 使用绝对路径
# model_path = r"D:\code\code_ml\work\DeiT-LT\checkpoint\deit_out_c10lt\deit_base_distilled_patch16_224_resnet32_200_CIFAR10LT_imb100_128_[deitlt_paco_sam_cifar10_if100]\deit_base_distilled_patch16_224_resnet32_200_CIFAR10LT_imb100_128_[deitlt_paco_sam_cifar10_if100]_checkpoint.pth"

# # 检查模型文件是否存在
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"Model file not found at {model_path}")

# # 调用函数
# extract_model_ranks(model_path)

