import json
import re
from tqdm import tqdm
import argparse


def isSimilar(text1, text2):
    dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]
    for i in range(1, len(text1) + 1):
        for j in range(1, len(text2) + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[-1][-1] / min(len(text1), len(text2)) >= 0.65 and abs(len(text1) - len(text2)) < max(len(text1),
                                                                                                   len(text2)) * 0.3


def process_ocr(ocr_list):
    flag = [False] * len(ocr_list)
    for i in range(len(ocr_list)):
        for j in range(i + 1, len(ocr_list)):
            if isSimilar(ocr_list[i], ocr_list[j]):
                flag[j] = True
    filtered_ocr_list = [ocr_list[i] for i in range(len(ocr_list)) if not flag[i]]
    return filtered_ocr_list


def isValid(text):
    chinese = ''.join(re.findall(r'[\u4e00-\u9fa5]', text))
    english = ''.join(re.findall(r'[A-Za-z]', text))
    number = ''.join(re.findall(r'[0-9]', text))
    #     print(number, '| ', chinese, '|', english)
    #     print(len(number), len(chinese), len(english))
    return len(number) < len(text) * 0.6 and (len(chinese) + len(english)) >= len(text) * 0.3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='labeled')
    args = parser.parse_args()
    
    filename_list = args.filename.split(';')
    for filename in filename_list:
        print('preprocessing ', filename)
        ### 无标注数据
        with open('data/annotations/{}.json'.format(filename), 'r') as fr:
            unlabeled_data = json.loads(fr.readline())

        print(len(unlabeled_data))

        for i in tqdm(range(len(unlabeled_data))):
            ocr_text = unlabeled_data[i]['ocr']
            ocr_text = [item['text'] for item in ocr_text if len(item['text']) > 0 and isValid(item['text'])]
            filtered_ocr = process_ocr(ocr_text)  # 过滤掉ocr相同或者大量相同的文本
            ocr_text = ';'.join(filtered_ocr)
            unlabeled_data[i]['ocr_processed'] = ocr_text

        with open('data/annotations/{}_processed.json'.format(filename), 'w') as fw:
            fw.write(json.dumps(unlabeled_data, ensure_ascii=False))