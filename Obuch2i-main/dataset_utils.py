import json
import pandas as pd 


def centroid(vertexes):
    _x_list = [vertex [0] for vertex in vertexes]
    _y_list = [vertex [1] for vertex in vertexes]
    _len = len(vertexes)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    return [_x, _y]


# Opening JSON files
f_test = open('ru_notebooks/annotations_test.json')
f_train = open('ru_notebooks/annotations_train.json')
f_val = open('ru_notebooks/annotations_val.json')
files = [f_test, f_train, f_val]
resulting = pd.DataFrame(columns=['id', 'file_name', 'text'])

for file in files:

    united_info = pd.DataFrame(columns=['id', 'file_name'])
    data = json.load(file)

    for i in data['images']:
        id = i['id']
        file_name = i['file_name']
        united_info.loc[len(united_info.index)] = [id, file_name] 
        f_test.close()

    cur = data['annotations'].copy()
    # df_positions = pd.DataFrame(columns=['id', 'text', 'polygon', 'centroid', 'x', 'y'])
    df_positions = pd.DataFrame(columns=['id', 'text'])
    for i in range(len(cur)):
        try:
            id = cur[i]['image_id']
            text = cur[i]['attributes']['translation']
            # x_list = []
            # y_list = []
            # for j in range(len(cur[i]['segmentation'][0])):
            #     if j % 2 == 0:
            #         x_list.append(cur[i]['segmentation'][0][j])
            #     else:
            #         y_list.append(cur[i]['segmentation'][0][j])
            # polygon = []
            # for j in range(len(x_list)):
            #     polygon.append(tuple([x_list[j], y_list[j]]))
            # df_positions.loc[len(df_positions.index)] = [id, text, tuple(polygon), None, None, None] 
            # df_positions['centroid'] = df_positions['polygon'].apply(centroid)
            # df_positions['x'] = df_positions['centroid'][0]
            # df_positions['y'] = df_positions['centroid'][1]
            df_positions.loc[len(df_positions.index)] = [int(id), text] 
        except KeyError:
            pass
    united_info = united_info.merge(df_positions, on='id')
    united_info = united_info.groupby('file_name')['text'].apply(list)
    frames = [resulting, united_info]
    resulting = pd.concat(frames)

resulting.to_csv('words_on_images.csv')