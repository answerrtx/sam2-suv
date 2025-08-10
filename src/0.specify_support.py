import openai
import base64
import os
#pip install openai==0.28
# 设置你的 API Key
openai.api_key ="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"

# 读取并编码图片
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 指定图片路径和提示语
image_path1 = "../Datasets_ONE/MSKUSO/support/imgs/sp_00005.jpg"
image_path2 = "../Datasets_ONE/MSKUSO/support/mask/sp_00005_mask.png"
image_path3 = "../Datasets_ONE/MSKUSO/support/mask/sp_00005_mask.png"
image_path4 = "../Datasets_ONE/MSKUSO/support/mask/sp_00005_mask.png"

#prompt_text = "The first is an ultrasound image; the second is a mask. The red mask indicates the UCL, the blue mask indicates the PrxPX, the green mask indicates the MC. Briefly describe its the transducer direction, class, color (not mask color), shape, and biologically mimetic features. Then summarize in a paragraph. When summrazing, don't include the mask color"
prompt_text = "The first is an ultrasound image; which of the following most likely to the first one"

# 编码图像为 base64
base64_image1 = encode_image(image_path1)
base64_image2 = encode_image(image_path2)

# 构建请求消息
response = openai.ChatCompletion.create(
    model="gpt-4.1",  # gpt-4o支持图像输入
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image1}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image2}"}}
        ]}
    ],
    temperature=0.7,
)

# 输出结果
print(response["choices"][0]["message"]["content"])

