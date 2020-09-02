from project2.zExplorerOpenCV import Explorer
import cv2

if __name__ == '__main__':
    video = 'D:/data/object2/has.mp4'
    video_capture = cv2.VideoCapture(0)
    explorer = Explorer(True)
    i = 0
    boxes = None
    # out = cv2.VideoWriter('2.avi')
    frames = []
    n = 0
    while True:
        success, img = video_capture.read()
        # img = cv2.resize(img, None, fx=0.4, fy=0.4)
        # print(img.shape)
        if success and (i % 2 == 0):
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = explorer.explore(image)
            # for box in boxes:
            #     x1 = box[0]
            #     y1 = box[1]
            #     x2 = box[2]
            #     y2 = box[3]
            #     w = x2 - x1
            #     h = y2 - y1
            #     c_x = x1 + w / 2-w*0.02
            #     c_y = y1 + h / 2
            #     sid_length = 0.4 * w*0.82
            #     c_x,c_y,sid_length = int(c_x),int(c_y),int(sid_length)
            #     x1 = c_x - sid_length
            #     y1 = c_y - sid_length
            #     x2 = c_x + sid_length
            #     y2 = c_y + sid_length
            #     try:
            #         if w > 112:
            #             image_crop = img[y1:y2,x1:x2]
            #             image_crop = cv2.resize(image_crop,(112,112))
            #             cv2.imwrite(f'D:/data/object2/faceimage/3/{n}.jpg', image_crop)
            #             n+=1
            #     except:
            #         pass
        for box in boxes:
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            # x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w = x2 - x1
            h = y2 - y1
            c_x = x1 + w / 2 - w * 0.02
            c_y = y1 + h / 2
            sid_length = 0.4 * w * 0.82
            c_x, c_y, sid_length = int(c_x), int(c_y), int(sid_length)
            x1 = c_x - sid_length
            y1 = c_y - sid_length
            x2 = c_x + sid_length
            y2 = c_y + sid_length
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('JK', img)
        i += 1
        if cv2.waitKey(1) == ord('q'):
            break
        # out.write(img)
        # frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    cv2.destroyAllWindows()