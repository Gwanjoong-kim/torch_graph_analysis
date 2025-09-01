import torch
import torch.fx as fx
import graphviz
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np
from torch.fx import Tracer
import inspect
import pdb

class MyTracer(Tracer):
    def to_bool(self, proxy):
        return False  # 무조건 False로 평가

local_path = "/home/audience/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0"

# Processor & Model 로드
__FULL_ORIG = torch.full

def __full_kw_compat(size, *args, **kwargs):
    # fill_value가 키워드로 들어온 경우 처리
    if "fill_value" in kwargs:
        fv = kwargs.pop("fill_value")
        return __FULL_ORIG(size, fv, *args, **kwargs)
    # fv가 bool 같은 특수 객체일 수 있으니 int 변환
    if len(args) >= 1 and isinstance(args[0], (bool, )):
        args = (int(args[0]),) + args[1:]
    return __FULL_ORIG(size, *args, **kwargs)

torch.full = __full_kw_compat

processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)

vla = AutoModelForVision2Seq.from_pretrained(
    local_path, trust_remote_code=True, device_map=None, attn_implementation="eager"
).to("cuda:0")

device = torch.device("cuda:0")
dtype  = next(vla.parameters()).dtype  # 보통 bfloat16/float16일 수 있음

# 더미 입력 생성
def random_image(width=224, height=224):
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, 'RGB')

image = random_image()
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

# processor가 실제 inference에서 쓰는 입력값 그대로 사용
inputs = processor(prompt, image, return_tensors="pt")

# ids/mask는 정수 유지, 픽셀은 모델 dtype로
inputs["input_ids"]      = inputs["input_ids"].to(device)
inputs["attention_mask"] = None
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(device=device, dtype=dtype)

inputs["past_key_values"] = None

# fx.trace에 넣을 concrete_args는 따로 만들지 않고, 모델 forward의 주요 입력만 지정
tracer = MyTracer()
graph = tracer.trace(vla, concrete_args=inputs)
traced_model = fx.GraphModule(vla, graph)

# Graphviz 시각화
graph = traced_model.graph
dot = graphviz.Digraph(comment='VLA FX Graph')

for node in graph.nodes:
    dot.node(node.name, f"{node.name}\nOp: {node.op}\nTarget: {node.target}")
    for arg in node.args:
        if isinstance(arg, fx.Node):
            dot.edge(arg.name, node.name)

dot.render('vla_model_fx_graph')