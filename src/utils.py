import json


def process_json(jsonfile):
    text_dict = []
    image_dict = {}
    data = json.load(open(jsonfile, 'r'))
    annotations = data['annotations']
    images = data['images']
    for image in images:
        image_dict[image['id']] = image['filename']
    for i in range(len(annotations)):
            annotations[i]['image_id'] = image_dict[annotations[i]['image_id']]
    return annotations



if __name__ == '__main__':
    process_json("/home/ubuntu/VQA/vqa/data/EVJVQA/evjvqa_train.json")