from pathlib import Path
import sys, torch, json
sys.path.insert(0,'/data2/ls/code/DiT-IC-c-sf-residual-20261025')
from models.bt_self_flow.experiment import load_config,make_model
from models.bt_self_flow.checkpoint import read_checkpoint
from models.bt_self_flow.data import ImageDataset,counter_seed
from models.bt_self_flow.losses import ImageLoss,reconstruction_objective
from models.bt_self_flow.packet import RateControl
from train_bt_self_flow import initialize_from
config=load_config('/data2/ls/code/DiT-IC-c-sf-residual-20261025/configs/bt_self_flow/quantization_residual/c_sf.json')
model=make_model(config,torch.device('cuda:0'))
initialize_from(model,read_checkpoint('/data2/ls/checkpoints/ditic-50k-warmups/c_precision_v2_lambda16_50k_20261022/backend_init.pt'),'joint')
model.train()
losses=ImageLoss(**config['image_loss']).cuda().eval()
data=ImageDataset(config['data']['train'],512)
image=torch.stack([data[(i,counter_seed(903,i,1101))] for i in range(8)]).cuda()
with torch.autograd.detect_anomaly(check_nan=True):
 with torch.autocast('cuda',dtype=torch.float16):
  result=model(image,RateControl(0.))
  loss,metrics=reconstruction_objective(result,image,RateControl(0.),model.distortion_scale.s_data,losses,config['objective'],config['quality'])
  print(json.dumps({k:float(v.detach()) for k,v in metrics.items()}),flush=True)
 loss.backward()
print('NONFINITE',[(n,int((~torch.isfinite(p.grad)).sum())) for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()],flush=True)
