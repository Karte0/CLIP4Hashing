import numpy as np

file_path = 'dataset/MSRVTT/caption'

ground_truth = np.zeros((1000, 1000))

for k in range(1000):
    file_name_now = f'video{k + 9000}.txt'
    full_path_now = file_path + '/' + file_name_now
    with open(full_path_now, "r", encoding='utf-8') as file_now:
        caption_now = []
        for line in file_now:
            line = line.strip()
            caption_now.append(line)
        #  print(caption_now)
        for i in range(1000):
            file_name = f'video{i + 9000}.txt'
            full_path = file_path + '/' + file_name
            with open(full_path, "r", encoding='utf-8') as file:
                caption = []
                for sentence in file:
                    caption.append(sentence.strip())
                if ground_truth[k][i] == 1:
                    break

                flag = 0
                for x in caption_now:
                    for y in caption:
                        if x in y or y in x:
                            ground_truth[k][i] = 1
                            ground_truth[i][k] = 1
                            flag = 1
                            print(f'{k} and {i} is related\n')
                            break
                    if flag == 1:
                        flag = 0
                        break

output_path = 'dataset/MSRVTT/ground_truth.txt'
np.savetxt(output_path, ground_truth, fmt='%d')