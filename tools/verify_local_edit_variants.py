"""Qwen 局部重绘的三个方案对比（计划 P0-2）：

  1. 原图（已跑过）：裁剪块 + cfg 4 → 脸被抹掉、指令「盔甲改翠绿」没执行
  2. A2 整图做条件 + 蒙版贴回：模型看得到完整角色，只把选区贴回去
  3. A3 裁剪块 + 低 CFG(1.5)：更贴近原图，指令权重下降

跑法：python tools/verify_local_edit_variants.py
产物：assets/_qwen_verify/a_variants.png
"""
import sys, os, time
sys.path.insert(0,'.')
os.environ['HF_HUB_OFFLINE']='1'
from PIL import Image, ImageDraw
import numpy as np
from workflows import repaint as rp
from workflows.engine import Engine
from workflows.generate import QWEN_NEG, _qwen_call

OUT='assets/_qwen_verify'
INSTR=("Edit this image: change the armor to bright emerald green, "
       "keep the character identity, face, hairstyle and background unchanged.")

def mask_of(src):
    m=Image.new('L',src.size,0); d=ImageDraw.Draw(m)
    w,h=src.size
    d.rectangle([int(w*0.34),int(h*0.22),int(w*0.66),int(h*0.46)],fill=255)
    return m

def report(tag, src, out, msk, dt):
    a=np.asarray(src.convert('RGB'),dtype=np.int16); b=np.asarray(out.convert('RGB'),dtype=np.int16)
    m=np.asarray(msk,dtype=np.uint8)>127
    out_max=float(np.abs(a-b)[~m].max()); in_mean=float(np.abs(a-b)[m].mean())
    print(f'[{tag}] {dt:.1f}s 选区外最大差={out_max:.0f} 选区内平均差={in_mean:.1f}',flush=True)

def main():
    eng=Engine('models')
    src=Image.open(f'{OUT}/char.png').convert('RGBA')
    msk=mask_of(src)

    # A2: 整图做条件（模型看得到完整角色）+ 蒙版贴回
    t=time.time()
    gen=_qwen_call(eng,prompt=INSTR,negative=QWEN_NEG,
                   ref_image=[src.convert('RGB')],width=768,height=768,steps=24,cfg=4.0,
                   seed=5,model='qwen-image-2.1',label='a2').convert('RGB')
    dt=time.time()-t
    if gen.size!=src.size: gen=gen.resize(src.size,Image.LANCZOS)
    out=rp.composite(src,gen,msk,feather_px=6)
    out.convert('RGB').save(f'{OUT}/a2_fullimg_mask.png')
    report('A2 整图+蒙版',src,out,msk,dt)

    # A3: 裁剪块 + 低 CFG（1.5）：更贴近原图，指令权重下降
    crop,cmask,box=rp.crop_region(src,msk,expand=8,context=0.35)
    t=time.time()
    gen=_qwen_call(eng,prompt=INSTR,negative=QWEN_NEG,
                   ref_image=[crop.convert('RGB')],width=768,height=768,steps=24,cfg=1.5,
                   seed=5,model='qwen-image-2.1',label='a3').convert('RGB')
    dt=time.time()-t
    if gen.size!=crop.size: gen=gen.resize(crop.size,Image.LANCZOS)
    genfull=src.convert('RGB').copy(); genfull.paste(gen,box[:2])
    out=rp.composite(src,genfull,msk,feather_px=6)
    out.convert('RGB').save(f'{OUT}/a3_crop_lowcfg.png')
    report('A3 裁剪块+cfg1.5',src,out,msk,dt)

    # 拼对比
    ims=[Image.open(p).convert('RGB').resize((320,320)) for p in
         [f'{OUT}/char.png',f'{OUT}/a_local_edit.png',f'{OUT}/a2_fullimg_mask.png',f'{OUT}/a3_crop_lowcfg.png']]
    sh=Image.new('RGB',(320*4,320),(20,20,24))
    for i,im in enumerate(ims): sh.paste(im,(320*i,0))
    sh.save(f'{OUT}/a_variants.png')
    print('对比图:', f'{OUT}/a_variants.png')
main()
