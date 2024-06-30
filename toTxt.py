import json

train_json_file_path = 'train_val_videodatainfo.json'
test_json_file_path = 'test_videodatainfo.json'

with open(train_json_file_path, 'r', encoding='utf-8') as file:
    train_data = json.load(file)

with open(test_json_file_path, 'r', encoding='utf-8') as file:
    test_data = json.load(file)

file_path = 'dataset/MSRVTT/caption'
for i in range(10000):
    txt = ''
    file_name = f'video{i}.txt'
    full_path = file_path + '/' + file_name
    for sentence in train_data['sentences']:
        now_id = sentence['video_id']
        if now_id == f'video{i}':
            txt = txt + sentence['caption'] + '\n'
    if txt != '':
        with open(full_path, "w", encoding='utf-8') as file:
            file.write(txt)
            print(f'{i}\n')
for i in range(3000):
    txt = ''
    file_name = f'video{i + 7010}.txt'
    full_path = file_path + '/' + file_name
    for sentence in test_data['sentences']:
        now_id = sentence['video_id']
        if now_id == f'video{i + 7010}':
            txt = txt + sentence['caption'] + '\n'
    if txt != '':
        with open(full_path, "w", encoding='utf-8') as file:
            file.write(txt)
            print(f'{i + 7010}\n')



