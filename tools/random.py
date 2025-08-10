import requests as rq

url = "https://www.random.org/integers/"
params = {
    "num": 5,         # 生成 5 个随机数
    "min": 1,         # 最小值
    "max": 100,       # 最大值
    "col": 1,         # 每列随机数数量
    "base": 10,       # 十进制
    "format": "plain",  # 输出为纯文本
    "rnd": "new"      # 每次生成新的随机数
}

response = rq.get(url, params=params)
if response.status_code == 200:
    print("Random Numbers:", response.text)
else:
    print("Error:", response.status_code)
