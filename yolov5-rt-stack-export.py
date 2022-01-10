import cv2
import torch

from yolort.utils import cv2_imshow, get_image_from_url, read_image_to_tensor
from yolort.utils.image_utils import color_list, plot_one_box
from yolort.models import yolov5s
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_raw = get_image_from_url("https://gitee.com/zhiqwang/yolov5-rt-stack/raw/master/test/assets/bus.jpg")
# img_raw = cv2.imread('../test/assets/bus.jpg')
img = read_image_to_tensor(img_raw)
img = img.to(device)
model = yolov5s(pretrained=True, score_thresh=0.55)
model.eval()
model = model.to(device)
model_out = model.predict(img)
print(f'Starting TorchScript export with torch {torch.__version__}...')
export_script_name = 'yolov5s.torchscript.pt'  # filename
model_script = torch.jit.script(model)
model_script.eval()
model_script = model_script.to(device)
# Save the scripted model file for subsequent use (Optional)
model_script.save(export_script_name)