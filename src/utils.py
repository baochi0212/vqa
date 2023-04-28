import json


def process_json(jsonfile):
    text_dict = []
    annotations = json.load(open(jsonfile, 'r'))['annotations']
    for i in range(len(annotations)):
        annotations[i]['image_id'] = f'0000000{annotations[i]["image_id"]}.jpg'
    return annotations




if __name__ == '__main__':
    process_json("/home/ubuntu/VQA/vqa/data/EVJVQA/evjvqa_train.json")