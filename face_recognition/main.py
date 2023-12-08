import os
import cv2
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from urllib import request

from mtcnn.mtcnn import MTCNN

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms

import sys
from pathlib import Path

from embeddings import get_embeddings
from face_crop import face_crop

BASE_PATH = 'C:/Users/yell0/Face_detection/Face-Recognition-with-ArcFace'

def run_face_recog():

    input_size = 112
    cos_similarity_threshold = 0.45
    
    crop_folder = os.path.join(BASE_PATH,'captured_faces/')
    result_folder = os.path.join(BASE_PATH,'result')

    # 기준 이미지 계산하기
    target_embeddings = []
    target_folder = os.path.join(BASE_PATH,'target/')

    
    _, target_embedding = get_embeddings(
        data_root=target_folder,
        model_root=os.path.join(BASE_PATH, "checkpoint/backbone_ir50_ms1m_epoch120.pth"),
        input_size=[input_size, input_size],
    )
    target_embeddings.append(target_embedding)

    # ----------------------------------------------------------
    #                      Step1. face detect
    # ----------------------------------------------------------
    print("Step1. face detect\n")

    output_folder = os.path.join(crop_folder, "user")
    if not os.path.exists(output_folder) :
        os.makedirs(output_folder)

    cap = cv2.VideoCapture('final.mp4') 

    detector = MTCNN()

    while True :
        ret, frame = cap.read()

        faces = detector.detect_faces(frame)

        # 감지된 얼굴에 사각형 그리고 캡처
        for i, face in enumerate(faces):
            x, y, width, height = face['box']
            x, y = max(0, x), max(0, y)  # 음수 좌표 방지
            roi = frame[y:y+height, x:x+width]
            cv2.imwrite(os.path.join(output_folder, f'user_{i}.jpg'), roi)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
    # ----------------------------------------------------------
    #                      Step2. embedding
    # ----------------------------------------------------------
        # 화면에 프레임 표시
        cv2.imshow('Target Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if len(faces) > 0:  # 얼굴이 감지되면 웹캠 종료.
            faces, embeddings = get_embeddings(
            data_root=crop_folder,
            model_root=os.path.join(BASE_PATH, "checkpoint/backbone_ir50_ms1m_epoch120.pth"),
            input_size=[input_size, input_size],
            )
    # ----------------------------------------------------------
    #                      Step3. cos_similarity 계산
    # ----------------------------------------------------------
            print("\n\n")
            print("Step3. cos_similarity 계산\n")

            for i, target_embedding in enumerate(target_embeddings[0]):
                similarity  = np.inner(target_embedding, embeddings)
                similarity = np.clip(similarity, 0, 1)

                if np.any(similarity > cos_similarity_threshold):
                    print(f"타겟 {i + 1} (이)랑 일치합니다.")
                    cv2.imwrite(os.path.join(result_folder, f'user_target_{i+1}.jpg'), roi)
                    break
                else :
                    continue
                print('타겟이랑 일치하지 않습니다.')
        
    
    # 웹캠 종료 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_recog()