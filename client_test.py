# client_test.py

import requests
import json
import yaml
import sys

# --- 加载配置以获取端口号 ---
with open('conf.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

PORT = config['server']['port']
API_URL = f"http://127.0.0.1:{PORT}/chat"

def run_chat_session():
    """
    启动一个交互式的命令行聊天会话
    """
    chat_history = []
    print("命令行聊天客户端已启动（流式输出）。输入 'exit' 或 'quit' 退出。")
    print("=" * 50)

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("会话结束。")
                break

            # 1. 将用户输入添加到历史记录
            chat_history.append({"role": "user", "content": user_input})
            
            # 2. 发送请求到后端（流式）
            payload = {"messages": chat_history}
            response = requests.post(API_URL, json=payload, stream=True, timeout=60)
            response.raise_for_status()

            # 3. 处理流式响应
            print("-" * 20)
            print("Assistant: ", end="", flush=True)
            
            full_response = ""
            debug_info = {}
            usage_info = {}
            
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])  # 去掉 "data: " 前缀
                        
                        if data.get("type") == "content":
                            # 处理内容块
                            choices = data.get("choices", [])
                            if choices and choices[0].get("delta", {}).get("content"):
                                content = choices[0]["delta"]["content"]
                                print(content, end="", flush=True)
                                full_response += content
                        
                        elif data.get("type") == "debug":
                            # 保存调试信息
                            debug_info = data.get("debug_info", {})
                        
                        elif data.get("type") == "done":
                            # 处理结束信号
                            usage_info = data.get("usage", {})
                            break
                        
                        elif data.get("type") == "error":
                            # 处理错误
                            error_msg = data.get("error", {}).get("message", "未知错误")
                            print(f"\n[错误] {error_msg}")
                            break
                            
                    except json.JSONDecodeError as e:
                        print(f"\n[警告] 无法解析JSON: {line}")
                        continue
            
            print()  # 换行
            print("-" * 20)
            
            # 打印调试信息
            if debug_info:
                retrieved_chunks = []
                
                # 合并两个知识库的检索结果
                if "security_knowledge_chunks" in debug_info:
                    retrieved_chunks.extend(debug_info["security_knowledge_chunks"])
                if "topsec_document_chunks" in debug_info:
                    retrieved_chunks.extend(debug_info["topsec_document_chunks"])
                
                # 打印检索到的知识，用于调试和验证
                if retrieved_chunks:
                    print("[DEBUG] 检索到的知识块:")
                    for i, chunk in enumerate(retrieved_chunks):
                        print(f"  {i+1}. {chunk[:100]}...") # 只打印前100个字符
                
                # 打印更详细的调试信息
                print(f"[DEBUG] 检索统计: 总文档{debug_info.get('total_retrieved_chunks', 0)}个, 对话轮数{debug_info.get('message_count', 0)}")
            
            # 打印使用统计信息
            if usage_info:
                print(f"[INFO] Token使用: {usage_info.get('total_tokens', 0)} (输入: {usage_info.get('prompt_tokens', 0)}, 输出: {usage_info.get('completion_tokens', 0)})")
            
            print("=" * 50)

            # 4. 将模型的回答也添加到历史记录
            if full_response:
                chat_history.append({"role": "assistant", "content": full_response})

        except requests.exceptions.RequestException as e:
            print(f"\n[错误] 无法连接到服务器: {e}")
            break
        except KeyboardInterrupt:
            print("\n\n用户中断，退出会话。")
            break
        except Exception as e:
            print(f"\n[错误] 发生未知错误: {e}")
            break

if __name__ == "__main__":
    run_chat_session()