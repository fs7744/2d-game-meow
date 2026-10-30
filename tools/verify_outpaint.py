import sys, os, time, torch
sys.path.insert(0,'.')
os.environ['HF_HUB_OFFLINE']='1'
from PIL import Image
from workflows.engine import Engine
from workflows import outpaint as op

eng=Engine('models'); pipe=eng.txt2img_pipe('qwen-image-2.1')

# 先生成一张有场景感的图（扩图要能看出"场景延续"才说明问题）
src_p = 'assets/_outpaint_src.png'
if not os.path.exists(src_p):
    t=time.time()
    im=pipe(prompt='a cozy forest clearing, mossy ground, tall trees, soft sunlight, '
                    '2D game background asset, wide shot',
            width=768, height=768, num_inference_steps=16, true_cfg_scale=1.0,
            generator=torch.Generator('cuda').manual_seed(7),
            output_resolution=768).images[0]
    im.convert('RGB').save(src_p)
    print(f'源图生成 {time.time()-t:.1f}s', flush=True)
src=Image.open(src_p).convert('RGBA')
print('源图', src.size, flush=True)

# 挂 Outpaint LoRA
eng.apply_loras(pipe, [{'name':'qwen21_outpaint','scale':0.9}])
print('LoRA:', eng.active_loras(), flush=True)

t=time.time()
r=op.outpaint(pipe, src=src, direction='right', expand=0.5, steps=8, cfg=1.0, seed=7)
dt=time.time()-t
r['image'].convert('RGB').save('assets/_outpaint_right.png')
print(f'扩图 {dt:.1f}s 画布={r["canvas"].size} work={r["work"]}', flush=True)
print('prompt:', r['prompt'][:110], flush=True)
