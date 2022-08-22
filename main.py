def script_method(fn, _rcb=None):
    return fn


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj

import numpy.core._dtype_ctypes
from torch import jit
script_method1 = jit.script_method
script1 = jit.script
jit.script_method = script_method
jit.script = script

from torch import LongTensor, no_grad
from commons import intersperse
from utils import get_hparams_from_file, load_checkpoint
from models import SynthesizerTrn
from symbols import symbols
from tkinter import END, Tk, Button, Label, Text, W, E, N, S
from sounddevice import play
from scipy.io.wavfile import write
from time import time, strftime, localtime
from os.path import exists
from tkinter.messagebox import showinfo

def cleaned_text_to_sequence(cleaned_text):
  _symbol_to_id = {s: i for i, s in enumerate(symbols)}
  sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
  return sequence

def get_text(text, hps):
    text_norm = cleaned_text_to_sequence(text)
    if hps.data.add_blank:
        text_norm = intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def tts(text):
  stn_tst = get_text(text, hps)
  with no_grad():
    x_tst = stn_tst.unsqueeze(0)
    x_tst_lengths = LongTensor([stn_tst.size(0)])
    global audio
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()
  play(audio, 22050)
def save():
  f = str(strftime('%Y%m%d%H%M%S.wav',localtime(time())))
  write(f, 22050, audio)
  if exists(f):
    showinfo(title='tip ',message=f"已保存为{f}")
  else:
    showinfo(title='tip ',message=f"失败")

#def create():
window = Tk()
#window.geometry('500x400')wavfile.write('recording.wav', fs, recording)

#var=StringVar()
Label(window, text='短句生成效果不佳').grid(row=0, column=1, stick=(N,S), pady=10)
e1 = Text(window, height = 10)
e1.grid(row=1, column=0, stick=E+W, columnspan=3)
b1=Button(window,text='生成',width=15,height=2,command=lambda:tts((e1.get('1.0',END)).replace('\n',' ').lower())).grid(row=2, column=0, stick=(N,S))
b1=Button(window,text='播放',width=15,height=2,command=lambda:play(audio, 22050)).grid(row=2, column=1, stick=(N,S))
b1=Button(window,text='保存',width=15,height=2,command=save).grid(row=2, column=2, stick=(N,S))
Label(window, text='只能输入包含!"&*,-.?ABCINU[ ]abcdefghijklmnoprstuwyz{ }~及空格的字符').grid(row=3, column=1, stick=(N,S), pady=10)
Label(window, text='@4454544').grid(row=4, column=1, stick=(N,S), pady=10)
hps = get_hparams_from_file("./configs/nan.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)
_ = net_g.eval()
_ = load_checkpoint("model.pth", net_g, None)

window.mainloop()