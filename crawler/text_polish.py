from openai import OpenAI
from transformers import set_seed
import json_repair
import time
import random
import os
import json

class GPTAgent:
    def __init__(self, model_name, system_prompt="你是一个专业的文本润色助手。你的任务是将输入文本润色成通顺、流畅的中文文本，去除特殊符号，保持原意。请直接返回润色后的文本，不要添加任何额外的解释或格式。", task="", max_new_tokens=512,
                 temperature=0.7, seed=10086, time_sleep_min=0.5, time_sleep_max=1.5):
        """
        初始化 GPTAgent
        """

        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.temperature = temperature
        self.model = OpenAI(
            base_url="",
            api_key="",
            timeout=300000
        )
        self.system_prompt = system_prompt
        self.seed = seed
        self.task = task
        self.time_sleep_min = time_sleep_min
        self.time_sleep_max = time_sleep_max
        set_seed(self.seed)

    def query(self, prompt, print_prompt=False):
        """
        调用 OpenAI API 并返回结果
        """
        prompt = prompt[:32768]
        if print_prompt:
            print(f"Prompt: {prompt}")

        sleep_time = random.randint(self.time_sleep_min, self.time_sleep_max)
        print(f"Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)

        try:
            completion = self.model.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )

            res = completion.choices[0].message.content
            if print_prompt:
                print(f"Response: {res}")

            return res

        except Exception as e:
            print(f"Error during API call: {e}")
            return None

    def polish_text(self, text):
        """
        对单个文本进行润色
        """
        prompt = f"请对以下文本进行润色，使其通顺流畅，去除特殊符号：\n\n{text}"
        result = self.query(prompt)
        if result is None:
            print("润色失败，返回原文")
            return text
        return result

    def polish_directory(self, input_dir, output_dir):
        """
        处理文件夹中的所有txt文件
        
        Args:
            input_dir: 输入文件夹路径
            output_dir: 输出文件夹路径，如果为None则使用input_dir
        """
            
        os.makedirs(output_dir, exist_ok=True)
        
        txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        
        for txt_file in txt_files:
            input_path = os.path.join(input_dir, txt_file)
            output_path = os.path.join(output_dir, f"polished_{txt_file}")
            
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                polished_content = self.polish_text(content)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(polished_content)
                    
                print(f"成功处理文件: {txt_file}")
                
            except Exception as e:
                print(f"处理文件 {txt_file} 时发生错误: {str(e)}")
                continue

if __name__ == "__main__":
    agent = GPTAgent(
        model_name="gpt-4o-mini",
        task="文本润色",
        max_new_tokens=512,
        temperature=0.7,
        seed=42,
        time_sleep_min=1,
        time_sleep_max=3,
    )

    input_directory = "output_1"
    output_directory = "output_2"
    
    agent.polish_directory(input_directory, output_directory)
