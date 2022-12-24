import os
import json
import yaml

from lxml import etree
from tqdm.auto import tqdm


def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args:
        xml: xml tree obtained by parsing XML file contents using lxml.etree
    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def get_voc_labels_from_yaml(yaml_dir):
    with open(yaml_dir, 'r') as f:
        labels_voc = yaml.safe_load(f)

    catid_labels_voc = labels_voc['labels']
    labels_catid_voc = {v:k for k,v in catid_labels_voc.items()}

    return labels_catid_voc


def xml2coco(ann_list, xml_dir, json_dir, labels_catid):
    """
    参数说明
    :param ann_list(list)：需要转换的xml文件列表。
    :param xml_dir(str)：xml文件的目录。
    :param json_dir(str)：转换后json文件保存的位置。
    :param labels_catid(dict)：以字典形式储存的目标类别编号，比如{'bus':0, 'person':1}。
    :return：转换后的文件会保存到json_dir位置。
    """

    data_json = {'images': [], 'annotations': [], 'categories': []}

    num = 0
    for n, i in tqdm(enumerate(ann_list)):
        xml_path = os.path.join(xml_dir, i)
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)["annotation"]
        img = {}
        num_objs = len(data['object'])
        img['id'] = n
        filename = data['filename']
        img['file_name'] = filename
        img['height'] = int(data['size']['height'])
        img['width'] = int(data['size']['width'])

        data_json['images'].append(img)

        for i in range(num_objs):
            ann = {'id': num, 'image_id': n}
            box = [
                int(data['object'][i]['bndbox']['xmin']), int(data['object'][i]['bndbox']['ymin']),
                int(data['object'][i]['bndbox']['xmax']), int(data['object'][i]['bndbox']['ymax'])
            ]
            ann['area'] = (box[2] - box[0]) * (box[3] - box[1])
            ann['bbox'] = [box[0], box[1], (box[2] - box[0]), (box[3] - box[1])]
            ann['category_id'] = labels_catid[data['object'][i]['name']] + 1
            ann['iscrowd'] = int(data['object'][i]['difficult'])
            data_json['annotations'].append(ann)
            num += 1

    for k, v in labels_catid.items():
        cat = {'id': v + 1, 'name': k}
        data_json['categories'].append(cat)

    with open(f'{json_dir}', 'w') as f:
        json.dump(data_json, f)

    print(f'ann_file has been saved in {json_dir}')


if __name__ == "__main__":

    """以测试集转换为例子"""

    # 获取目标类别，用于最后的格式转换
    labels_voc = get_voc_labels_from_yaml('../voc/labels_voc.yaml')

    # 获取测试集文件名，这里用voc2007作为测试集，测试集文件信息记录在test.txt里，因此需要从中读取
    test_list = []
    with open('./voc2007/test.txt') as f:
        for i in f:
            test_list.append(i.strip() + '.xml')

    # 测试集目标框文件的目录
    xml_dir = './voc2007/Annotations'

    # 转换后文件的保存地址
    json_dir = './voc2007/test_ann.json'

    # 进行转换
    xml2coco(test_list, xml_dir, json_dir, labels_voc)
