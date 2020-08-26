from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np
import urllib.request
import boto3  # Amazon's aws library for python 3
import requests
from io import BytesIO
import cv2
# import av
from matplotlib import cm

dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()


def decode_segmap(image, nc=21):

    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128,
                                                        0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64,
                                                              0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0,
                                                           128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    print(rgb)
    return rgb


def seg():
    net = dlab
    # video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
    cap = cv2.VideoCapture(
        "https://d390v2huy8d28s.cloudfront.net/3000450/1291084/3000450-1291084.m3u8")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out2 = cv2.VideoWriter('your_video.avi', fourcc, 40.0, size)
    count = 0
    while(cap.isOpened()):
        count = count + 1
        _, frame = cap.read()
        # cv2.imshow('Recording...', frame)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('red.png', frame)
        # name = "frame%d.jpg"%count
        # cv2.imwrite(name, frame)
        # img = Image.fromarray('red.png')
        # img = Image.open("red.png")
        # img = Image.fromarray(np.uint8(cm.gist_earth(frame)*255))
        img = Image.fromarray(frame.astype('uint8'), 'RGB')
        print(img, "image PIL")
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()
        trf = T.Compose([
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
        inp = trf(img).unsqueeze(0)
        # print(inp)
        out = net(inp)['out']
        print(out)
        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        print(om)
        rgb = decode_segmap(om)
        print(rgb)

        # plt.imshow(rgb)

        # plt.axis('off')
        # plt.savefig('fig.png')
        # im = cv2.imread('fig.png')
        # img_cv = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        out2.write(rgb)

        plt.show()
        # plt.savefig('horse.png')

    cap.release()
    out.release()
    cv2.destroyAllWindows()


seg()


def video_segmentation_with_av():
    container = av.open(video_url, timeout=60)
    stream = container.streams.video[0]
    stream.codec_context.skip_frame = 'NONKEY'
    frame_count = 0
    frame_time = None
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (240, 426))
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out2 = cv2.VideoWriter('your_video.avi', fourcc, 20.0, size)
    for n, frame in enumerate(container.decode(stream)):
        # fourcc = 'mp4v'
        # vid_writer = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*fourcc), 15 , (w, h))
        print(frame)
        frame_count += 1
        if frame.pts == None:
            continue
        # frame_time = frame.pts * stream.time_base * 1000
        # frame_times.append(frame_time)
        img = frame.to_image()
        print(img)
        img_cv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w, layers = img_cv.shape

        # fourcc = 'mp4v'
        # vid_writer = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*fourcc), 15 , (w, h))
        # cv2.imshow('Recording...', img_cv)
        vid_writer.write(img_cv)
        print(h, w)
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()
        # vid_writer.release()
        # vid_writer.write(img_cv)

    vid_writer.release()
