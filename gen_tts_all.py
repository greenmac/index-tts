from indextts.infer_v2 import IndexTTS2
from utils_tool import timer
import opencc
import torch
import numpy as np
import sounddevice as sd

@timer
def export_tts_all():
    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml", 
        model_dir="checkpoints", 
        use_fp16=True, 
        device="cuda:0",
        # use_cuda_kernel=True, 
        # use_deepspeed=True,
    )
    
    text_ori = "现在闲家胜率高达百分之86, 赶紧加码下注吧!"
    text = get_text_cn(text_ori)
    
    # get_all_gen_tts_functions(tts, text) # 一次生成並播放全部
    get_segmented_gen_tts_functions(tts, text) # 分段生成並播放
    
def get_text_cn(text_ori) -> str:
    # 't2s' = 繁體轉簡體 (字對字)
    # 'tw2sp' = 台灣繁體轉大陸簡體 (包含慣用語轉換，例如 滑鼠->鼠标)
    converter = opencc.OpenCC('t2s') 
    return converter.convert(text_ori)

def play_audio(audio_data, sample_rate=24000):
    """
    智慧型播放函數：能處理 Tuple (sr, audio) 或單純的 Tensor/Numpy
    """
    final_audio = None
    final_sr = sample_rate

    if isinstance(audio_data, (tuple, list)):
        for item in audio_data:
            if isinstance(item, (torch.Tensor, np.ndarray)):
                final_audio = item
            elif isinstance(item, int):
                final_sr = item # 使用模型回傳的採樣率
    elif isinstance(audio_data, (torch.Tensor, np.ndarray)):
        final_audio = audio_data

    if final_audio is None:
        print("播放失敗: 無法識別音訊數據格式", type(audio_data))
        return
    
    if isinstance(final_audio, torch.Tensor):
        final_audio = final_audio.squeeze().cpu().numpy()
    
    final_audio = final_audio.astype(np.float32)

    if final_audio.ndim > 1 and final_audio.shape[0] == 1:
        final_audio = final_audio.flatten()

    # 4. 播放
    # print(f"正在播放 (SR={final_sr})...")
    final_audio = final_audio / 32768.0
    sd.play(final_audio, samplerate=final_sr)
    sd.wait()

def get_once_gen_tts_functions(tts: IndexTTS2, text: str):
    audio = tts.infer(
        spk_audio_prompt='./data/trump/train/train_trump_01.wav', 
        text=text, 
        output_path=None
        # output_path="gen_tts_all_once.wav"
    )
    play_audio(audio, sample_rate=24000)
    
def get_segmented_gen_tts_functions(tts: IndexTTS2, text: str):
    chunks = text.split(", ")
    for idx, chunk in enumerate(chunks):
        # print(f">> 生成並保存片段: {chunk}")
        # output_path = f"gen_{chunk[:5]}.wav"
        audio = tts.infer(
            spk_audio_prompt='./data/trump/train/train_trump_01.wav', 
            text=chunk, 
            # output_path=None
            output_path=f"gen_tts_all_segment_{idx+1}.wav"
        )
        play_audio(audio, sample_rate=24000)

if __name__ == "__main__":
    export_tts_all()