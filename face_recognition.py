import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import time
import os

# Kích thước khung hình
frame_size = (640, 480)
DATA_PATH = './data'

# Chuẩn hóa ảnh đầu vào cho mô hình
def trans(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    return transform(img)

# Tải embeddings và tên từ file
def load_faceslist(device):
    filename = 'faceslistCPU.pth' if device == 'cpu' else 'faceslist.pth'
    embeds = torch.load(os.path.join(DATA_PATH, filename), map_location=device)
    names = np.load(os.path.join(DATA_PATH, 'usernames.npy'))
    return embeds, names

# Cắt ảnh khuôn mặt từ frame gốc
def extract_face(box, img, margin=20):
    face_size = 160
    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin),
    ]
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, frame_size[0])),
        int(min(box[3] + margin[1] / 2, frame_size[1])),
    ]
    cropped = img[box[1]:box[3], box[0]:box[2]]
    resized_face = cv2.resize(cropped, (face_size, face_size), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized_face)

# Nhận dạng khuôn mặt
def inference(model, face, local_embeds, threshold=3):
    with torch.no_grad():
        face_embedding = model(trans(face).to(device).unsqueeze(0))
        diff = face_embedding.unsqueeze(-1) - local_embeds.T.unsqueeze(0)
        norm_score = torch.sum(diff ** 2, dim=1)  # (1, n)
        min_dist, embed_idx = torch.min(norm_score, dim=1)
        if min_dist.item() > threshold:
            return -1, -1
        else:
            return embed_idx.item(), min_dist.item()

# === Chạy chương trình chính ===
if __name__ == "__main__":
    prev_frame_time = 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Tải mô hình nhận dạng
    model = InceptionResnetV1(pretrained="casia-webface", classify=False).to(device)
    model.eval()

    # Khởi tạo MTCNN
    mtcnn = MTCNN(keep_all=True, thresholds=[0.7, 0.7, 0.8], device=device)

    # Tải dữ liệu khuôn mặt
    embeddings, names = load_faceslist(device)

    # Mở webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                bbox = list(map(int, box.tolist()))
                face = extract_face(bbox, frame)
                idx, score = inference(model, face, embeddings)

                if idx != -1:
                    label = f"{names[idx]}_{score:.2f}"
                    color = (0, 255, 0)  # Xanh lá: đã nhận diện
                else:
                    label = "Unknown"
                    color = (0, 0, 255)  # Đỏ: người lạ

                # Vẽ khung và nhãn
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                frame = cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1, cv2.LINE_AA)

        # Tính và hiển thị FPS
        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time + 1e-5))
        prev_frame_time = new_frame_time
        cv2.putText(frame, f'FPS: {fps}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)

        # Hiển thị frame
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
            break

    cap.release()
    cv2.destroyAllWindows()
