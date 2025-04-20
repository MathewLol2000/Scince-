import base64
import requests
import json
from collections import Counter


def encode_file(file_name):
  """
  Method used for encoding images to base64 encoding
      Parameters: 
          file_name - name of the input image file 
      Returns: 
          image encoded in base64
  """
  with open(file_name, "rb") as f:
    encoded_image = base64.b64encode(f.read())
  return encoded_image


def api_request(IAM_TOKEN, YA_DIR_ID, body_json, file_name):
   """
   Method used for making api-request to yandex cloud OCR(text recognition) model 
      Parameters: 
          IAM_TOKEN - yandex IAM Token, accesed via '$yc iam create-token' in linux
          YA_DIR_ID - id of the yandex cloud folder (not cloud!)
          body_json - json-file for the api-request, created via method create_body_json 
          file_name - name of the output file 
      Returns: 
          None
   """
   headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {IAM_TOKEN}',
    'x-folder-id': f'{YA_DIR_ID}',
    'x-data-logging-enabled': 'true',
    }
   with open(body_json) as f:
     data = f.read().replace('\n', '').replace('\r', '').encode()
    
   response = requests.post('https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText', headers=headers, data=data)
   with open(f'jsons/output_{file_name.split("/")[2].split(".")[0]}.json', 'wb') as f:
     f.write(response.content)
     f.close()


def create_body_json(encoded_img):
  """
   Method used for creating body.json
      Parameters: 
          encoded_img - image encoded in base64
      Returns: 
          None
   """
  with open('jsons/body.json', 'w') as f:
    text = f'{{ \n \t "mimeType": "JPEG", \n \
    "languageCodes": ["ru","en"], \n \
    "model": "handwritten", \n \
    "content": "{encoded_img.decode("utf-8")}"}}'
    f.write(text)
    f.close()

def json_to_text(file_name):
  """
   Method used for converting json to text
      Parameters: 
          i - counter from the cycle, as method used in for-loop
      Returns: 
          None
   """
  f = open(f'jsons/output_{file_name.split("/")[2].split(".")[0]}.json')

    # returns JSON object as 
    # a dictionary
  data = json.load(f)

    # Iterating through the json
    # list
  res = """"""
  for j in data["result"]['textAnnotation']['fullText']:
     res += j
  with open(f'txts/res_texts/res_text_{file_name.split("/")[2].split(".")[0]}.txt', 'w') as f:
    f.write(res)
    f.close()

def execute_text_from_img(IAM_TOKEN, YA_DIR_ID, body_json, FILENAME):
  """
   Method used for extracting text information from resulted JSON 
      Parameters: 
          IAM_TOKEN - yandex IAM Token, accesed via '$yc iam create-token' in linux
          YA_DIR_ID - id of the yandex cloud folder (not cloud!)
          body_json - json-file for the api-request, created via method create_body_json
          i - counter from the cycle, as method used in for-loop
          
      Returns: 
          None
   """
  data = encode_file(FILENAME)
  create_body_json(data)
  api_request(IAM_TOKEN, YA_DIR_ID, body_json, FILENAME)


def levenshtein_distance(s, t):
    """
    Method for calculating levenstein distance between two strings
        Parameters:
            s - string 1
            t - string 2
        
        Returns: 
            Levenstein distance between strings
    """
    m, n = len(s), len(t)
    if m < n:
        s, t = t, s
        m, n = n, m
    d = [list(range(n + 1))] + [[i] + [0] * n for i in range(1, m + 1)]
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s[i - 1] == t[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1
    return d[m][n]
 
def compute_similarity(input_string, reference_string):
    """
    Method for calculating similarity between 2 strings
    """
    distance = levenshtein_distance(input_string, reference_string)
    max_length = max(len(input_string), len(reference_string))
    similarity = 1 - (distance / max_length)
    return similarity

def number_of_duplicates(a, b):
  return len(set(a).intersection(set(b))) / max(len(set(a)), len(set(b)))