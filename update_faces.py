import glob
import torch 
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import os
from PIL import Image
import numpy as np

IMG_PATH = ' data\test_images\kiet'
DATA_PATH = './data'

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

def trans(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    return transform(img)

# Kiem tra thu muc ton tai
if not os.path.exists(IMG_PATH):
    print(f"Loi: Khong tim thay thu muc anh tai {IMG_PATH}")
    exit()

model = InceptionResnetV1(
    classify=False,
    pretrained="casia-webface"
).to(device)
model.eval()

embeddings = []
names = []

for usr in os.listdir(IMG_PATH):
    user_path = os.path.join(IMG_PATH, usr)
    if not os.path.isdir(user_path):
        continue
    embeds = []
    for file in glob.glob(os.path.join(user_path, '*.jpg')):
        try:
            img = Image.open(file)
            img_tensor = trans(img).to(device).unsqueeze(0)
            with torch.no_grad():
                embeds.append(model(img_tensor))  # [1, 512]
        except Exception as e:
            print(f"Loi khi xu ly anh {file}: {e}")
            continue
    if len(embeds) == 0:
        print(f"Khong co anh hop le cho nguoi dung: {usr}")
        continue
    embedding = torch.cat(embeds).mean(0, keepdim=True)
    embeddings.append(embedding)
    names.append(usr)

# Neu khong co anh nao
if len(embeddings) == 0:
    print("Khong tim thay anh nao de tao embeddings. Thoat.")
    exit()

embeddings = torch.cat(embeddings)  # [n, 512]
names = np.array(names)

if device == 'cpu':
    torch.save(embeddings, os.path.join(DATA_PATH, "faceslistCPU.pth"))
else:
    torch.save(embeddings, os.path.join(DATA_PATH, "faceslist.pth"))

np.save(os.path.join(DATA_PATH, "usernames"), names)
print(f' Update OK! Co {names.shape[0]} nguoi trong danh sach.')
