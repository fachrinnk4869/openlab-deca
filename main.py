# save as send_deca.py
import socket
import json
from decalib.deca import DECA
from torchvision import transforms
from PIL import Image
import cv2
import torch
from decalib.utils.config import cfg as deca_cfg
import numpy as np
from torch.utils.data import Dataset, DataLoader
from glob import glob
from skimage.transform import estimate_transform, warp
from facenet_pytorch import MTCNN  # for face detection
np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.complex = np.complex_
np.object = np.object_
np.unicode = np.unicode_
np.str = np.str_


class FAN(object):
    def __init__(self):
        import face_alignment
        self.model = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, flip_input=False)

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        if out is None:
            return [0], 'kpt68'
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:, 0])
            right = np.max(kpt[:, 0])
            top = np.min(kpt[:, 1])
            bottom = np.max(kpt[:, 1])
            bbox = [left, top, right, bottom]
            return bbox, 'kpt68'


class TestData():
    def __init__(self, iscrop=True, crop_size=224, scale=1.25, face_detector='fan', sample_step=10):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        # print('total {} images'.format(len(self.imagepath_list)))
        # self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        self.face_detector = FAN()

    def __len__(self):
        return len(self.imagepath_list)

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type == 'kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0,
                              bottom - (bottom - top) / 2.0])
        elif type == 'bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0,
                              bottom - (bottom - top) / 2.0 + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def run(self, image):
        # imagepath = self.imagepath_list[index]
        # imagename = os.path.splitext(os.path.split(imagepath)[-1])[0]
        image = np.array(image)
        # image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        h, w, _ = image.shape
        if self.iscrop:
            bbox, bbox_type = self.face_detector.run(image)
            if len(bbox) < 4:
                print('no face detected! run original image')
                left = 0
                right = h-1
                top = 0
                bottom = w-1
            else:
                left = bbox[0]
                right = bbox[2]
                top = bbox[1]
                bottom = bbox[3]
            old_size, center = self.bbox2point(
                left, right, top, bottom, type=bbox_type)
            size = int(old_size*self.scale)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] -
                               size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        else:
            src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])

        DST_PTS = np.array(
            [[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image/255.

        dst_image = warp(image, tform.inverse, output_shape=(
            self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2, 0, 1)
        return {'image': torch.tensor(dst_image).float(),
                # 'imagename': imagename,
                'tform': torch.tensor(tform.params).float(),
                'original_image': torch.tensor(image.transpose(2, 0, 1)).float(),
                }


# Setup DECA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
deca_cfg.model.use_tex = True
# deca_cfg.rasterizer_type = args.rasterizer_type
deca_cfg.model.extract_tex = True
deca = DECA(config=deca_cfg).to(device)

cap = cv2.VideoCapture(0)

detector = TestData(iscrop=True, crop_size=224,
                    scale=1.25, face_detector='fan')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and crop face
    # img_cropped = mtcnn(frame)
    testdata = detector.run(frame)
    img_cropped = testdata['image'].to(device)
    # print(f"Image shape: {img_cropped.shape}")
    if img_cropped is None:
        continue  # no face detected
    img_t = img_cropped[None, ...]  # Convert to tensor if needed
    # img_t = transform(transforms.ToPILImage()(
    #     img_cropped)).unsqueeze(0).to(device)

    with torch.no_grad():
        codedict = deca.encode(img_t)
        exp_params = codedict['exp'].cpu().numpy().flatten().tolist()
        opdict, visdict = deca.decode(codedict)  # tensor
        # if True:
        #     tform = testdata['tform'][None, ...]
        #     tform = torch.inverse(tform).transpose(1, 2).to(device)
        #     original_image = testdata['original_image'][None, ...].to(
        #         device)
        #     _, orig_visdict = deca.decode(
        #         codedict, render_orig=True, original_image=original_image, tform=tform)
        #     orig_visdict['inputs'] = original_image
    cv2.imshow('DECA keypoint', deca.visualize(visdict))
    # Map to blendshape-like naming
    # blend_data = {f"Exp{i}": float(v) for i, v in enumerate(exp_params)}
    # print(f"Blendshape data: {blend_data}")
    # sock.sendto(json.dumps(blend_data).encode(), (SERVER_IP, SERVER_PORT))
    cv2.imshow('DECA Live', img_cropped.permute(1, 2, 0).cpu().numpy())
    # cv2.imshow('DECA Live', img_cropped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
