import requests
import time
from openai import OpenAI
from dotenv import dotenv_values
import os, re
import httpx

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
config = dotenv_values(env_path)


import tiktoken, random
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

random.seed(10086)


def token_length(text):
    return len(encoding.encode(text, disallowed_special=()))


MAX_LENGTH = 65536
def get_answer(content, system_prompt=None, model='deepseek-v4-flash'):

    reasoning_output = ""
    output = ''
    api_key = config.get("DEEPSEEK_API_KEY")
    base_url = config.get("DEEPSEEK_BASE_URL")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    client = OpenAI(api_key=api_key, base_url=base_url, http_client=httpx.Client(verify=False))


    if token_length(content) > MAX_LENGTH:
        content = content[:MAX_LENGTH]
        print(f"Input text truncated to {MAX_LENGTH} tokens.")

    extra_kwargs = {
        "stream": False,
        "extra_body": {"thinking": {"type": "enabled"}},
        "reasoning_effort": "high",
    }

    if system_prompt is not None:
        response = client.chat.completions.create(
            model=model,
            temperature=0.6,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            **extra_kwargs,
        )

    else:
        response = client.chat.completions.create(
            model=model,
            # temperature=0.6,
            messages=[
                {"role": "user", "content": content},
            ],
            **extra_kwargs,
        )

    for i in range(3):
        try:
            #官网
            reasoning_output = response.choices[0].message.reasoning_content.strip()
            output = response.choices[0].message.content.strip()

            #官网流式
            # for chunk in response:
            #     if chunk.choices[0].delta.reasoning_content:
            #         reasoning_output += chunk.choices[0].delta.reasoning_content
            #     else:
            #         output += chunk.choices[0].delta.content
            # print("---------------------------思考过程---------------------------")
            # print(reasoning_output)
            # print("\n---------------------------最终答案:---------------------------")
            # print(output)

            if '<answer>' in output:
                print(1)
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

def parse_answer_r1(answer):
    cleaned_answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    return cleaned_answer
if __name__ == '__main__':
    ans = get_answer("计算圆周率是多少")
    
    # ans = parse_answer_r1(ans)
    print(ans)

