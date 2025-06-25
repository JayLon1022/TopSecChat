from openai import OpenAI
from transformers import set_seed
import json_repair
import time
import random
import os
import json
import shutil

class GPTAgent:
    def __init__(self, model_name, system_prompt="你是一个文件命名助手。你的任务是根据文件内容，返回文件合适的命名，不要添加任何额外的解释或格式。", task="", max_new_tokens=512,
                 temperature=0.7, seed=10086, time_sleep_min=0.5, time_sleep_max=1.5):
        """
        初始化 GPTAgent
        """

        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.temperature = temperature
        self.model = OpenAI(
            base_url="https://xiaoai.plus/v1",
            api_key="sk-GYvK0yJXWFkz6ADkyJKviUPnOjKoolYTLv1RH4DfQMyUwrTK",
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


    def rename_file(self, old_name, file_content=None, file_path=None):
        """
        使用大模型为单个文件生成新名称
        
        Args:
            old_name: 原文件名
            file_content: 文件内容（可选）
            file_path: 文件路径（可选）
        
        Returns:
            str: 新的文件名（不包含路径）
        """
        prompt_parts = [f"请为以下文件生成一个有意义的中文文件名：\n原文件名：{old_name}"]
        
        if file_content:
            # 截取前1000个字符作为参考
            content_preview = file_content[:1000] + "..." if len(file_content) > 1000 else file_content
            prompt_parts.append(f"文件内容预览：\n{content_preview}")
        
        if file_path:
            prompt_parts.append(f"文件路径：{file_path}")
        
        prompt_parts.append("\n要求：\n1. 文件名要简洁明了，能反映文件内容\n2. 只返回文件名，不要包含路径或扩展名\n3. 使用中文命名\n4. 避免使用特殊字符")
        
        prompt = "\n".join(prompt_parts)
        
        result = self.query(prompt)
        if result is None:
            print(f"重命名失败，保持原文件名: {old_name}")
            return old_name
        
        new_name = result.strip()

        new_name = os.path.basename(new_name)

        name_without_ext = os.path.splitext(new_name)[0]
        
        # 获取原文件的扩展名
        _, ext = os.path.splitext(old_name)
        
        return name_without_ext + ext

    def rename_directory(self, input_dir, output_dir=None, read_content=False, pdf_dir=None, pdf_output_dir=None):
        """
        批量重命名文件夹中的txt文件，同时重命名对应的pdf文件
        
        Args:
            input_dir: 输入文件夹路径（txt文件所在目录）
            output_dir: 输出文件夹路径，如果为None则使用input_dir
            read_content: 是否读取文件内容来辅助命名
            pdf_dir: pdf文件所在目录，如果为None则不处理pdf文件
            pdf_output_dir: pdf文件输出目录，如果为None则使用pdf_dir
        """
        if output_dir is None:
            output_dir = input_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        if pdf_dir and pdf_output_dir:
            os.makedirs(pdf_output_dir, exist_ok=True)
        
        if not os.path.exists(input_dir):
            print(f"错误：输入目录不存在: {input_dir}")
            return
        
        if pdf_dir and not os.path.exists(pdf_dir):
            print(f"错误：PDF目录不存在: {pdf_dir}")
            return
        
        all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith('.txt')]     
        print(f"找到 {len(all_files)} 个txt文件需要重命名")
        
        for filename in all_files:
            input_path = os.path.join(input_dir, filename)
            
            try:
                file_content = None
                if read_content:
                    try:
                        with open(input_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                    except UnicodeDecodeError:
                        try:
                            with open(input_path, 'r', encoding='gbk') as f:
                                file_content = f.read()
                        except:
                            print(f"无法读取文件内容: {filename}")
                            file_content = None
                
                new_filename = self.rename_file(filename, file_content, input_path)
                
                output_path = os.path.join(output_dir, new_filename)
                
                # 如果新文件名已存在，添加数字后缀
                counter = 1
                original_new_filename = new_filename
                while os.path.exists(output_path):
                    name_without_ext, ext = os.path.splitext(original_new_filename)
                    new_filename = f"{name_without_ext}_{counter}{ext}"
                    output_path = os.path.join(output_dir, new_filename)
                    counter += 1
                
                if input_dir == output_dir:
                    os.rename(input_path, output_path)
                else:
                    shutil.copy2(input_path, output_path)
                
                print(f"重命名txt文件成功: {filename} -> {new_filename}")
                
                if pdf_dir:
                    # txt文件名格式为: polished_xxx.txt，对应的pdf文件名为: xxx.pdf
                    if filename.startswith('polished_'):
                        pdf_filename = filename[9:]
                        pdf_filename = pdf_filename.replace('.txt', '.pdf')
                        
                        pdf_input_path = os.path.join(pdf_dir, pdf_filename)
                        
                        if os.path.exists(pdf_input_path):
                            new_pdf_filename = os.path.splitext(new_filename)[0] + '.pdf'
                            pdf_output_path = os.path.join(pdf_output_dir or pdf_dir, new_pdf_filename)
                            
                            pdf_counter = 1
                            original_pdf_filename = new_pdf_filename
                            while os.path.exists(pdf_output_path):
                                name_without_ext, ext = os.path.splitext(original_pdf_filename)
                                new_pdf_filename = f"{name_without_ext}_{pdf_counter}{ext}"
                                pdf_output_path = os.path.join(pdf_output_dir or pdf_dir, new_pdf_filename)
                                pdf_counter += 1
                            
                            if pdf_dir == (pdf_output_dir or pdf_dir):
                                os.rename(pdf_input_path, pdf_output_path)
                            else:
                                shutil.copy2(pdf_input_path, pdf_output_path)
                            
                            print(f"重命名pdf文件成功: {pdf_filename} -> {new_pdf_filename}")
                        else:
                            print(f"警告：找不到对应的pdf文件: {pdf_filename}")
                    else:
                        print(f"警告：txt文件名不符合预期格式（应以'polished_'开头）: {filename}")
                
            except Exception as e:
                print(f"重命名文件 {filename} 时发生错误: {str(e)}")
                continue

if __name__ == "__main__":
    rename_agent = GPTAgent(
        model_name="gpt-4o-mini",
        system_prompt="你是一个专业的文件命名助手。你的任务是根据文件内容或原文件名，为文件生成有意义的中文文件名。请直接返回文件名，不要添加任何额外的解释或格式。",
        task="文件重命名",
        max_new_tokens=256,
        temperature=0.7,
        seed=42,
        time_sleep_min=1,
        time_sleep_max=3,
    )


    # 配置路径
    input_directory = "../security_knowledge"
    output_directory = "../security_knowledge_rename"
    pdf_directory = "../original_pdf" 
    pdf_output_directory = "../security_pdf_rename" 
    
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    print(f"输入目录: {input_directory}")
    print(f"输出目录: {output_directory}")
    print(f"PDF目录: {pdf_directory}")
    print(f"PDF输出目录: {pdf_output_directory}")
    
    # 检查路径
    if not os.path.exists(input_directory):
        print(f"错误：输入目录不存在: {input_directory}")
        print("请检查路径是否正确，或者使用绝对路径")
        exit(1)
    if not os.path.exists(pdf_directory):
        print(f"错误：PDF目录不存在: {pdf_directory}")
        print("请检查路径是否正确，或者使用绝对路径")
        exit(1)
    
    task_type = "rename"
    
    print("开始文件重命名任务...")
    rename_agent.rename_directory(
        input_directory, 
        output_directory, 
        read_content=True,
        pdf_dir=pdf_directory,
        pdf_output_dir=pdf_output_directory
    )
