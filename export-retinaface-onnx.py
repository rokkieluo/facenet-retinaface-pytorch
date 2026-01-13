from nets_retinaface.retinaface import RetinaFace
from utils.config import cfg_mnet
import torch
model_path='model_data/Retinaface_mobilenet0.25.pth' #模型路径
model=RetinaFace(cfg=cfg_mnet) #模型初始化
device = torch.device('cpu')
model.load_state_dict(torch.load(model_path,map_location=device),strict=False) #模型加载
net=model.eval()
example=torch.rand(1,3,640,640) #给定输入
torch.onnx.export(model,(example),'model_data/retinaface.onnx',verbose=True,opset_version=9) #导出