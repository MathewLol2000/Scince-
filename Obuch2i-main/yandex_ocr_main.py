from utils import json_to_text, execute_text_from_img, compute_similarity, number_of_duplicates
import time
import os
import pandas as pd
import ast 
import json
import seaborn as sns 
import numpy as np
from scipy import stats
from preprocessing import remove_noise, erosion, get_grayscale, histogram_equalization, hsv, lines_removal
import cv2
# '$yc iam create-token' - should be refreshed each 12 hours 
IAM_TOKEN = 't1.9euelZrOi8mVnYmazsyNx5XOzovLmu3rnpWalMbHlsvKipLIyJaalZSPjczl9Pc7fVZK-e8TdwrZ3fT3eytUSvnvE3cK2c3n9euelZqRj8eSk4_Mz4mbmc7Klpicju_8xeuelZqRj8eSk4_Mz4mbmc7Klpicjg.H6vL1L9Y1I3FgMFQpgS7QcGV1tXasDxCRV8KY14BOyqRiYeTPi7uOJxq9ZQWO9kq-JJ1n2ga-rOcxKuFp-zZDA'
# copy this from yandex cloud - should be placed near name of catalog 
YA_DIR_ID = 'b1ghkpruumbocmh45sf2'
BODY_JSON = 'jsons/body.json'
BOOL_RUN_API = True
BOOL_RUN_MEASUREMENT = True
FILTERS = True




if BOOL_RUN_API:
    cnt = 0
    # send images for recognition to yandex cloud
    for filename in os.scandir('content_ai_copy/images'):
        if cnt < 100:
            image = cv2.imread(filename.path)
            if FILTERS:
                # replace here with filter 
                # image = remove_noise(image)
                # image = erosion(image)
                image = hsv(image)
                cv2.imwrite(filename.path, image)

            execute_text_from_img(IAM_TOKEN, YA_DIR_ID, BODY_JSON, filename.path)
            json_to_text(filename.path)
            time.sleep(1)
            cnt += 1


dist = []
if BOOL_RUN_MEASUREMENT:
    data = pd.read_csv('words_on_images.csv',sep=';', on_bad_lines='skip')
    data['file_name'] = data['file_name'].apply(lambda x: x.split('.')[0])
    # validate the quality for each image
    for filename in os.scandir('txts/res_texts'):

        with open(f'txts/res_texts/{filename.path.split("/")[2].split(".")[0]}.txt', 'r') as file:
            res_text =  file.read().replace('\n', ' ')
        # print(f'{filename.path.split("/")[2].split(".")[0].split("res_text_")[1]}.jpg')
        data_cur = data[data['file_name'] == f'{filename.path.split("/")[2].split(".")[0].split("res_text_")[1]}']
        try:
            text_true = ast.literal_eval(list(data_cur['text'])[0])
            res_text = res_text.split(' ')
            cnt = number_of_duplicates(text_true, res_text)
            dist.append(cnt)
        except Exception:
            pass

print(dist)
print(np.mean(dist))
print(stats.mode(dist))
print(np.median(dist))

# # OLD VERSION 
# cur_path = '/Users/ilyakasimov/Documents/text_recognition/images/raw_images'
# if BOOL_RUN_API:
#     cnt = 0
#     # send images for recognition to yandex cloud
#     for i in range(11):
#         if FILTERS:
#             image = cv2.imread(f'{cur_path}/img{i}.jpg')
#                 # replace here with filter 
#             # image = erosion(image)
#             image = get_grayscale(image)
#             # image = histogram_equalization(image)
#             cv2.imwrite(filename.path, image)

#         execute_text_from_img(IAM_TOKEN, YA_DIR_ID, BODY_JSON, filename.path)
#         json_to_text(filename.path)
#         time.sleep(1)
#         cnt += 1

# dist = []

# if BOOL_RUN_MEASUREMENT:
#     # validate the quality for each image
#     for i in range(1, 11):
#         with open(f'txts/true_texts/true_text_{i}.txt', 'r') as file:
#             true_text =  file.read().replace('\n', ' ')
        
#         with open(f'content_ai/res_texts_clean/res_text_{i}.txt', 'r') as file:
#             res_text =  file.read().replace('\n', ' ')
#         score = compute_similarity(res_text, true_text)
#         print(f'score of Levenstein on example number {i}: \t', score)


