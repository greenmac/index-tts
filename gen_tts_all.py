from indextts.infer_v2 import IndexTTS2
from utils_tool import timer

@timer
def export_tts_all():
    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml", 
        model_dir="checkpoints", 
        use_fp16=True, 
        device="cuda:0",
        use_cuda_kernel=True, 
        use_deepspeed=True,
    )
    text = "现在闲家胜率高达百分之86, 赶紧加码下注吧!"

    tts.infer(spk_audio_prompt='./data/trump/train/train_trump_01.wav', text=text, output_path="gen.wav", verbose=True)

if __name__ == "__main__":
    export_tts_all()