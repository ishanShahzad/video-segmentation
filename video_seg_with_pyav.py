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
