import cv2
import os
import sys
import time
import shutil
'''
    video data pre-processing tool.
        divide video to frames
        divide frames to sets
        convert frames to video
'''

def get_data(input_path):
    """Parse the data from annotation file

    Args:
        input_path: annotation file path

    Returns:
        all_data: list(filepath, width, height, list(bboxes))
        classes_count: dict{key:class_name, value:count_num}
            e.g. {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745}
        class_mapping: dict{key:class_name, value: idx}
            e.g. {'Car': 0, 'Mobile phone': 1, 'Person': 2}
    """
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    i = 1

    with open(input_path, 'r') as f:

        print('Parsing annotation files')

        for line in f:

            # Print process
            sys.stdout.write('\r' + 'idx=' + str(i))
            i += 1

            # line_split = line.strip().split(',')

            # (filename, y1, x1, y2, x2, class_name) = line_split
            print(line)
            filename = line.split()[0]
            y1, x1, y2, x2, class_name = line.split()[1].split(',')

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print(
                        'Found class name bg.Will be treated as background (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}

                # img = cv2.imread(filename)
                # (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                # all_imgs[filename]['width'] = cols
                # all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []
            # if np.random.randint(0,6) > 0:
            # 	all_imgs[filename]['imageset'] = 'trainval'
            # else:
            # 	all_imgs[filename]['imageset'] = 'test'

            all_imgs[filename]['bboxes'].append(
                {'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping


def video_to_imgs(folder_path, name, out_path):
    # video to frames
    path = os.path.join(folder_path, name)
    video = cv2.VideoCapture(path)
    # success, image = video.read()
    count = 1
    success = True
    while success:
        success, image = video.read()
        if success is not True:
            break
        cv2.imwrite("{}{}{}.jpg".format(out_path, name.strip('.avi'), count),
                image)
        if cv2.waitKey(10) == 27:
            break
        count += 1
    return count


def frame_to_video(frame_path, out_path, video_name, fps=16):
    # frames to video
    filelist = os.listdir(frame_path)
    for i in range(len(filelist)):
        filelist[i] = int(filelist[i][len(video_name):].strip('.jpg'))
    # fps = 12
    filelist.sort()
    for i in range(len(filelist)):
        filelist[i] = video_name+str(filelist[i])+'.jpg'
    file_path = os.path.join(out_path, video_name) + ".mp4"
    # fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
    size = cv2.imread(os.path.join(frame_path, filelist[0])).shape[:2]
    # notice:frame size should be (width, height)
    video = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]))
    for item in filelist:
        if item:
            item = os.path.join(frame_path, item)
            img = cv2.imread(item)
            video.write(img)
    video.release()


def divide_data(test_file, img_path, new_img_path):
    # divide images into image sets
    test_data, _, _ = get_data(test_file)
    test_imgs = dict()
    for test_img in test_data:
        test_imgs[os.path.basename(test_img['filepath'])] = test_img
    imgs = os.listdir(img_path)
    for img in imgs:
        try:
            # move test imgs to new folder
            test_imgs[img]
            shutil.move(os.path.join(img_path, img), os.path.join(new_img_path, img))
        except:
            continue
    return 0


# ----- config ----- #
if __name__ == '__main__':
    # divide video to frames
    folder_path = './Videos/'
    out_path = './Images/'
    pro_video_idx = list()        # list of video

    videos = os.listdir(folder_path)
    print(videos)

    # divide video to frames
    for idx in pro_video_idx:
        imgs = video_to_imgs(folder_path, videos[idx], out_path)

    # divide frames to different sets
    test_file = './test.txt'
    img_path = './Images'
    new_path = './Images_test'
    divide_data(test_file, img_path, new_path)

    # frames to video
    frame_path = ''
    out_path = ''
    frame_to_video(frame_path, out_path, '')

