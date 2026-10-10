import requests
import time
import base64
from dotenv import dotenv_values
import os
import mimetypes
import importlib.util
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
config = dotenv_values(env_path)
collect_image_inputs = None


def _collect_image_inputs(payload, *, max_images=12):
    global collect_image_inputs
    if collect_image_inputs is None:
        multimodal_path = Path(current_dir).parent / "multimodal.py"
        spec = importlib.util.spec_from_file_location("docpilot_multimodal", multimodal_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load multimodal helpers from {multimodal_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        collect_image_inputs = module.collect_image_inputs
    return collect_image_inputs(payload, max_images=max_images)



# qwen2.5-max
def get_answer(content, image=None, images=None, system_prompt=None, model='qwen-plus'):

    output = ''

    api_url = config.get("DASHSCOPE_URL")
    api_key = config.get("DASHSCOPE_API")
    # api_url = config.get("GPT_URL_PRO")
    # api_key = config.get("GPT_KEY_PRO")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=api_url)
    
    extra_kwargs = {
        "stream": False,
        "extra_body": {"thinking": {"type": "enabled"}},
        # "reasoning_effort": "high",
    }


    image_inputs = _collect_image_inputs({"image": image, "images": images}, max_images=12)
    user_content = _content_with_images(content, image_inputs)

    if system_prompt is not None:
        response = client.chat.completions.create(
            model=model,
            # temperature=0.6,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            **extra_kwargs,
        )
    else:
        response = client.chat.completions.create(
            model=model,
            # temperature=0.6,
            messages=[
                {"role": "user", "content": user_content},
            ],
            # stream=False
            **extra_kwargs,

        ) 

    for i in range(3):
        try:
            reasoning_output = response.choices[0].message.reasoning_content.strip()
            # print(reasoning_output)
            output = response.choices[0].message.content.strip()
            if '<answer>' in output:
                final_response = f"""
                <think>
                {reasoning_output}
                </think>
                {output}
                """
            else:
                final_response = f"""
                <think>
                {reasoning_output}
                </think>
                <answer>
                {output}
                </answer>
                """
            return final_response
        except:
            time.sleep(5)

    return output


def _content_with_images(text, images):
    if not images:
        return text
    parts = [{"type": "text", "text": str(text)}]
    for image in images:
        parts.append({"type": "image_url", "image_url": {"url": _image_url(image)}})
    return parts


def _image_url(value):
    if value.startswith(("http://", "https://", "data:image/")):
        return value
    path = Path(value).expanduser()
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"

import os
import asyncio
import platform


api_url = config.get("DASHSCOPE_URL")
api_key = config.get("DASHSCOPE_API")
_async_client = None


def _get_async_client():
    global _async_client
    if _async_client is None:
        from openai import AsyncOpenAI

        _async_client = AsyncOpenAI(api_key=api_key, base_url=api_url)
    return _async_client

# 定义异步任务列表
async def task(question):
    print(f"Sending question: {question}")
    response = await _get_async_client().chat.completions.create(
        messages=[
            {"role": "user", "content": question}
        ],
        model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    )
    print(f"Received answer: {response.choices[0].message.content}")

# 主异步函数
async def main():
    questions = ["你是谁？", "你会什么？", "天气怎么样？"]
    tasks = [task(q) for q in questions]
    await asyncio.gather(*tasks)

    
    
if __name__ == '__main__':
    answer = get_answer("你是什么模型", model='qwen3.7-plus')
    print(answer)

    # # 设置事件循环策略
    # if platform.system() == 'Windows':
    #     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # # 运行主协程
    # asyncio.run(main(), debug=False)
