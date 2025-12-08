from data.voice_gen_text import gen_text_list
from indextts.infer_v2 import IndexTTS2
from pathlib import Path
from utils_tool import timer
from utils_tool import get_ensure_dir
import opencc
import torch
import numpy as np
import sounddevice as sd

@timer
def export_tts_whole(category: str, spk_audio_prompt: Path, text_path: Path, emo_audio_prompt: str, emo_alpha: float):
    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml", 
        model_dir="checkpoints", 
        use_fp16=True, 
        device="cuda:0",
        # use_cuda_kernel=True, 
        # use_deepspeed=True,
    )
    
    for item in gen_text_list:
        host = item['host']
        context = item['context']
        if host == category:
            for entry in context: # entry 表示每個語音內容條目
                title: str = entry['title']
                text_ori = entry['text']
                text = get_text_cn(text_ori)
                output_path = get_ensure_dir(spk_audio_prompt.parent / title.rsplit('_', 1)[0]) /f'{category}_voice_{title}.wav'
            
                get_once_gen_tts_functions(spk_audio_prompt, tts, text, output_path, emo_audio_prompt, emo_alpha) # 一次生成並播放全部
                # get_segmented_gen_tts_functions(spk_audio_prompt, tts, text, output_path, emo_audio_prompt, emo_alpha) # 分段生成並播放
    
    
    # # text_ori = "The player's winning percentage is very high right now, so hurry up and increase your bet!"
    # text_ori = text_path.read_text(encoding='utf-8')
    # text = get_text_cn(text_ori)
    # output_path = spk_audio_prompt.parent / f'{category}_{text_path.stem}.wav'
    
    # get_segmented_gen_tts_functions(spk_audio_prompt, tts, text, output_path, emo_audio_prompt, emo_alpha) # 分段生成並播放
    
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

def get_once_gen_tts_functions(spk_audio_prompt: Path, tts: IndexTTS2, text: str, output_path: Path, emo_audio_prompt: str, emo_alpha: float):
    audio = tts.infer(
        spk_audio_prompt=str(spk_audio_prompt), 
        text=text, 
        # output_path=None
        output_path=str(output_path),
        emo_audio_prompt=emo_audio_prompt, # 情感音頻範例
        emo_alpha=emo_alpha, # 情感強度調整參數 (1.0 為原始情感強度，數值越大情感越明顯)
    )
    # play_audio(audio, sample_rate=24000)
    
def get_segmented_gen_tts_functions(spk_audio_prompt: Path, tts: IndexTTS2, text: str, output_path: Path, emo_audio_prompt: str, emo_alpha: float):
    chunks = text.split(", ")
    for idx, chunk in enumerate(chunks):
        # print(f">> 生成並保存片段: {chunk}")
        # output_path = f"gen_{chunk[:5]}.wav"
        audio = tts.infer(
            spk_audio_prompt=str(spk_audio_prompt), 
            text=chunk, 
            # output_path=None
            output_path=str(output_path.parent / f"{output_path.stem}_segment_{idx+1}.wav"),
            emo_audio_prompt=emo_audio_prompt, # 情感音頻範例
            emo_alpha=emo_alpha, # 情感強度調整參數 (1.0 為原始情感強度，數值越大情感越明顯)
        )
        play_audio(audio, sample_rate=24000)

if __name__ == "__main__":
    root_dir = Path('./data')
    category = 'Trump'
    # category = 'JackyChen'
    # category = 'PenélopeCruz'
    train_dir = root_dir / category / 'train'
    
    spk_audio_prompt = train_dir / f'train_{category}_01.wav'
    text_path = train_dir / 'Voice_EncourageBetting_gen.txt'
    emo_audio_prompt = './data/Trump/train/train_Trump_01.wav'
    emo_alpha = 3.0
    export_tts_whole(category, spk_audio_prompt, text_path, emo_audio_prompt, emo_alpha)