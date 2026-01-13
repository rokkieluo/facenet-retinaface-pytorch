from nets.facenet import Facenet
from torch import onnx
import torch 

model_path='model_data/facenet_mobilenet.pth' #模型路径 
model = Facenet(backbone="mobilenet",mode="predict") #模型初始化
device = torch.device('cpu')
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
example=torch.rand(1,3,160,160) #给定一个输入
torch.onnx.export(model,example,'model_data/facenet.onnx',verbose=True,opset_version=9) #导出
