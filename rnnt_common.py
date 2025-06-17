"""
RNNT 模型训练和评估的公共配置参数
"""
import torch


class Config:
    """RNNT模型配置类"""
    
    # 训练参数
    epochs = 50
    batch_size = 12
    lr = 0.0001
    hidden_dim = 256
    accum_steps = 1
    grad_clip = 1.0
    
    # 流式训练参数
    streaming = True
    static_chunk_size = 32
    use_dynamic_chunk = True
    num_decoding_left_chunks = 6
    
    # 损失函数权重
    ctc_weight = 0.3
    
    # 设备和路径配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = "/root/tf-logs/speech-recognition-rnnt"
    save_dir = "./models"
    
    # 评估参数
    eval_model_path = "./model.pt"
    eval_dataset = "dev"
    eval_output = None  # 结果输出文件路径
    use_ctc = True  # 是否使用CTC解码
    
    # 数据路径
    train_wav_scp = "./dataset/split/train/wav.scp"
    train_text = "./dataset/split/train/pinyin"
    test_wav_scp = "./dataset/split/test/wav.scp"
    test_text = "./dataset/split/test/pinyin"
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 50)
        print("RNNT 模型配置:")
        print("=" * 50)
        print(f"训练轮数: {cls.epochs}")
        print(f"批大小: {cls.batch_size}")
        print(f"学习率: {cls.lr}")
        print(f"隐藏层维度: {cls.hidden_dim}")
        print(f"梯度累积步数: {cls.accum_steps}")
        print(f"梯度裁剪阈值: {cls.grad_clip}")
        print()
        print(f"流式训练: {cls.streaming}")
        if cls.streaming:
            print(f"静态块大小: {cls.static_chunk_size}")
            print(f"使用动态块: {cls.use_dynamic_chunk}")
            print(f"左侧块数: {cls.num_decoding_left_chunks}")
        print()
        print(f"CTC权重: {cls.ctc_weight}")
        print(f"RNNT权重: {1.0 - cls.ctc_weight}")
        print(f"使用设备: {cls.device}")
        print("=" * 50)
