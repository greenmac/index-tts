import os
import torch
import sounddevice as sd
import numpy as np
from indextts.infer_v2 import IndexTTS2
from utils_tool import timer

@timer
def export_tts_stream():
    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml", 
        model_dir="checkpoints", 
        use_fp16=True, 
        device="cuda:0",
        use_cuda_kernel=True, 
        use_deepspeed=True,
    )
    # text = "现在闲家胜率高达百分之86, 赶紧加码下注吧!"
    text = (
        "现在闲家胜率高达百分之86, 赶紧加码下注吧! "
        "不仅如此，我们的系统显示目前的走势非常有利于连续下注，"
        "请各位玩家把握机会，不要错过这次千载难逢的好时机，"
        "祝大家好运连连，赢得盆满钵满！"
    )
    

    spk_audio_prompt = "./data/trump/train/train_trump_01.wav" 

    if not os.path.exists(spk_audio_prompt):
        print(f"警告: 找不到 {spk_audio_prompt}，嘗試使用 examples/voice_01.wav")
        spk_audio_prompt = "examples/voice_01.wav"

    # stream_return=True 是關鍵，這會讓它變成一個可以迭代的物件
    generator = tts.infer_generator(
        spk_audio_prompt=spk_audio_prompt, 
        text=text, 
        output_path=None,  # 串流模式下，這裡設為 None 也可以，或者給一個路徑最後存檔用
        stream_return=True,
        verbose=True
    )

    print(f">> 開始生成並播放: {text}")
    sample_rate = 22050  # IndexTTS 預設通常是 22050，若聲音變快或變慢請調整此數值 (例如 24000)
    
    try:
        for output_chunk in generator:
            if isinstance(output_chunk, torch.Tensor):
                audio_data = output_chunk.squeeze().cpu().float().numpy()
                audio_data = audio_data / 32768.0
                sd.play(audio_data, samplerate=sample_rate, blocking=True)
                
            elif isinstance(output_chunk, str):
                print(f">> 檔案已保存至: {output_chunk}")
                
    except Exception as e:
        print(f">> 播放過程中發生錯誤: {e}")

    print(">> 播放結束")
    
if __name__ == "__main__":
    export_tts_stream()