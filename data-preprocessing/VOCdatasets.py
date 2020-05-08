import xml.etree.cElementTree as ET
'''
    Train data pre-processing tool for VOC dataset.
'''

def xml_data_pro(file_path, data_path):
    """
        read img bbox data from one xml doc

    return:

        bboxes_pra : string of format 'filename y1, x1, y2, x2, class'

    """
    tree = ET.ElementTree(file=file_path)
    root = tree.getroot()
    file_name = root.find('filename').text
    bboxes = list(root.findall('object'))
    bboxes_pra = []
    for bbox in bboxes:
        box = bbox.find('bndbox')
        cls = bbox.find('name').text
        x1 = box.find('xmin').text
        y1 = box.find('ymin').text
        x2 = box.find('xmax').text
        y2 = box.find('ymax').text
        bboxes_coord = [y1, x1, y2, x2, cls]
        bboxes_coord = ','.join(bboxes_coord)
        bboxes_coord = data_path + '/' + file_name + ' '+bboxes_coord
        bboxes_pra.append(bboxes_coord)
    return bboxes_pra
    # print(bboxes_pra)


def folder_pro(folder_path, data_path, file_list, out_path):
    """
        process all xml doc in one folder
    :param folder_path:
    :param file_list: file list of img to process
    :param out_path:
    :return: .txt file in out_path
    """
    with open(out_path, 'w') as output:
        with open(file_list, 'r') as f:
            file_list = f.readlines()
        f.close()
        for name in file_list:
            path = folder_path + '/' + name.strip('\n') + '.xml'
            bboxes = xml_data_pro(path, data_path)
            for bbox in bboxes:
                output.write(bbox + '\n')


# ------------------------- Config -------------------------------- #
if __name__ == '__main__':
    # note that there should be no empty line in the end of train/test list file
    folder_path = "./xml"       # xml file path
    file_list = "./train.txt"       # train.txt
    output_path = "./train.txt"     # output file path
    data_path = "./data/"        # jpeg data path to util.py

    # ------------------------- Train ---------------------------- #
    folder_pro(folder_path, data_path, file_list, output_path)
    # val data
    file_list = "./VOC2007/train/ImageSets/Main/val.txt"
    output_path = "./VOC2007/val.txt"
    folder_pro(folder_path, data_path, file_list, output_path)

    # ------------------------- Test ----------------------------- #
    folder_path = "./VOC2007/test/Annotations"
    file_list = "./VOC2007/test/ImageSets/Main/test.txt"
    output_path = "./VOC2007/test.txt"
    data_path = "./data/VOC2007/test/JPEGImages"
    folder_pro(folder_path, data_path, file_list, output_path)


