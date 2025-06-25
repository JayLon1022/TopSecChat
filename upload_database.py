# manage_index.py

import os
import yaml
import logging
import sys
import fitz  # PyMuPDF

# 配置日志
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    Document,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- 1. 全局配置 ---

def load_config():
    """加载配置文件"""
    with open('conf.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_services():
    """配置LlamaIndex的全局服务 (Embedding模型)"""
    print("🚀 Configuring services...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5",
        cache_folder="./models_cache"
    )
    print("✅ Services configured.")

# MODIFIED: 在元数据中同时添加 file_name 和 file_path
def get_file_metadata(file_path: str) -> dict:
    """为文档创建元数据，同时包含文件路径、文件名和修改时间"""
    return {
        "file_path": file_path,                       # 保留完整路径，用于同步逻辑
        "file_name": os.path.basename(file_path),     # <<<---【核心修改】新增：文件名，用于前端显示
        "last_modified": os.path.getmtime(file_path)
    }

def is_pdf_parseable(pdf_path: str, min_text_ratio: float = 0.1) -> bool:
    """检测PDF是否适合解析（非图片型PDF）"""
    try:
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            doc.close()
            return False
        
        # 采样前3页或全部页面（如果少于3页）
        sample_pages = min(3, len(doc))
        total_chars = 0
        total_images = 0
        
        for page_num in range(sample_pages):
            try:
                page = doc[page_num]
                text = page.get_text()
                total_chars += len(text.strip())
                # 尝试获取图片，如果失败就跳过图片计数
                try:
                    total_images += len(page.get_images())
                except:
                    pass
            except Exception as page_error:
                print(f"⚠️ Error processing page {page_num} in {pdf_path}: {page_error}")
                continue
        
        doc.close()
        
        # 如果文本太少且图片很多，认为是图片型PDF
        if total_chars < 100 and total_images > sample_pages * 2:
            return False
            
        # 如果平均每页文本字符数太少，可能是扫描件
        if sample_pages > 0:
            avg_chars_per_page = total_chars / sample_pages
            if avg_chars_per_page < 50:
                return False
        
        # 如果总字符数太少，不值得处理
        if total_chars < 20:
            return False
            
        return True
        
    except Exception as e:
        print(f"⚠️ Error checking PDF {pdf_path}: {e}")
        return False

def extract_pdf_text(pdf_path: str) -> str:
    """使用PyMuPDF提取PDF文本，带有错误恢复机制"""
    text = ""
    doc = None
    
    try:
        # 尝试正常打开PDF
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                page_text = page.get_text()
                text += page_text + "\n"
            except Exception as page_error:
                print(f"⚠️ Error extracting text from page {page_num} in {pdf_path}: {page_error}")
                # 尝试使用其他文本提取方法
                try:
                    page_text = page.get_text("text")  # 明确指定文本模式
                    text += page_text + "\n"
                except:
                    # 如果还是失败，尝试字典模式
                    try:
                        text_dict = page.get_text("dict")
                        page_text = ""
                        for block in text_dict.get("blocks", []):
                            if "lines" in block:
                                for line in block["lines"]:
                                    for span in line.get("spans", []):
                                        page_text += span.get("text", "")
                        text += page_text + "\n"
                    except:
                        print(f"⚠️ Completely failed to extract text from page {page_num}")
                        continue
        
        doc.close()
        
    except Exception as e:
        print(f"⚠️ Error opening PDF {pdf_path}: {e}")
        if doc:
            try:
                doc.close()
            except:
                pass
        return ""
    
    return text.strip()

def load_documents_with_local_parsing(knowledge_base_path: str):
    """使用本地解析加载文档"""
    documents = []
    skipped_pdfs = []
    processed_count = 0
    
    print(f"📂 Scanning directory: {knowledge_base_path}")
    
    for dirpath, _, filenames in os.walk(knowledge_base_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            
            if filename.lower().endswith('.pdf'):
                processed_count += 1
                print(f"🔍 Processing PDF ({processed_count}): {filename}")
                
                # 检查PDF是否适合解析
                if not is_pdf_parseable(file_path):
                    skipped_pdfs.append(file_path)
                    print(f"⏭️ Skipping image-heavy/low-text PDF: {filename}")
                    continue
                
                # 提取PDF文本
                text = extract_pdf_text(file_path)
                if text.strip():
                    metadata = get_file_metadata(file_path)
                    doc = Document(text=text, metadata=metadata)
                    documents.append(doc)
                    print(f"✅ Successfully processed: {filename} ({len(text)} chars)")
                else:
                    skipped_pdfs.append(file_path)
                    print(f"⏭️ Skipping PDF with no extractable text: {filename}")
                    
            elif filename.lower().endswith(('.txt', '.md')):
                # 处理文本文件
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    if text.strip():
                        metadata = get_file_metadata(file_path)
                        doc = Document(text=text, metadata=metadata)
                        documents.append(doc)
                        print(f"✅ Successfully processed: {filename}")
                except Exception as e:
                    print(f"⚠️ Error reading {file_path}: {e}")
    
    if skipped_pdfs:
        print(f"📋 Skipped {len(skipped_pdfs)} PDFs (image-heavy or unreadable)")
        print("Skipped files:")
        for skipped in skipped_pdfs[:10]:  # 只显示前10个
            print(f"  - {os.path.basename(skipped)}")
        if len(skipped_pdfs) > 10:
            print(f"  ... and {len(skipped_pdfs) - 10} more")
    
    print(f"📊 Successfully loaded {len(documents)} documents")
    return documents

# --- 2. 核心索引管理逻辑 ---

def create_new_index(knowledge_base_path: str, storage_path: str):
    """首次创建全新的索引，使用本地PDF解析"""
    print(f"🔎 No existing index found. Starting fresh indexing from '{knowledge_base_path}'...")
    
    if not os.path.exists(knowledge_base_path) or not os.listdir(knowledge_base_path):
        os.makedirs(knowledge_base_path, exist_ok=True)
        print(f"❌ Knowledge base directory is empty. Please add documents to '{knowledge_base_path}'.")
        return

    docs = load_documents_with_local_parsing(knowledge_base_path)

    if not docs:
        print("🤷 No processable documents found in the knowledge base.")
        return

    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    index.storage_context.persist(persist_dir=storage_path)
    print(f"🎉 New index created successfully with {len(docs)} documents.")

def synchronize_index(knowledge_base_path: str, storage_path: str):
    """同步现有索引，执行增删改操作，使用本地PDF解析"""
    print(f"🔄 Found existing index. Synchronizing with '{knowledge_base_path}'...")
    
    storage_context = StorageContext.from_defaults(persist_dir=storage_path)
    index = load_index_from_storage(storage_context)
    
    # 获取索引中的文件信息，使用规范化路径
    indexed_files = {}
    print(f"📋 Current indexed documents:")
    for node_id, node in index.docstore.docs.items():
        if node.metadata and "file_path" in node.metadata:
            file_path = os.path.normpath(node.metadata["file_path"])
            indexed_files[file_path] = node.metadata.get("last_modified", 0)
            print(f"  - {os.path.basename(file_path)} (path: {file_path})")
    
    # 获取磁盘文件信息，使用规范化路径
    disk_files = {}
    print(f"📂 Current disk files:")
    for dirpath, _, filenames in os.walk(knowledge_base_path):
        for filename in filenames:
            if filename.lower().endswith(('.pdf', '.txt', '.md')):
                file_path = os.path.normpath(os.path.join(dirpath, filename))
                disk_files[file_path] = get_file_metadata(file_path)["last_modified"]
                print(f"  - {filename} (path: {file_path})")

    added_files = disk_files.keys() - indexed_files.keys()
    deleted_files = indexed_files.keys() - disk_files.keys()
    updated_files = {path for path, mtime in disk_files.items() if path in indexed_files and mtime > indexed_files[path]}

    print(f"📊 Analysis:")
    print(f"  - Added files: {len(added_files)}")
    print(f"  - Deleted files: {len(deleted_files)}")
    print(f"  - Updated files: {len(updated_files)}")

    if not added_files and not deleted_files and not updated_files:
        print("✅ Index is already up to date.")
        return

    changes_made = False

    # 处理删除的文件 - 使用ref_doc_id进行删除
    for file_path in deleted_files:
        print(f"🗑️ Deleting: {os.path.basename(file_path)}")
        try:
            # 找到对应的文档节点
            nodes_to_delete = []
            for node_id, node in index.docstore.docs.items():
                if node.metadata and node.metadata.get("file_path") == file_path:
                    nodes_to_delete.append(node_id)
            
            # 删除找到的节点
            for node_id in nodes_to_delete:
                index.delete_ref_doc(node_id, delete_from_docstore=True)
                print(f"  Deleted node: {node_id}")
            
            if nodes_to_delete:
                changes_made = True
            else:
                print(f"  ⚠️ No nodes found for {file_path}")
        except Exception as e:
            print(f"  ⚠️ Error deleting {file_path}: {e}")

    # 处理更新的文件
    for file_path in updated_files:
        print(f"🔄 Updating: {os.path.basename(file_path)}")
        try:
            # 先删除旧版本
            nodes_to_delete = []
            for node_id, node in index.docstore.docs.items():
                if node.metadata and node.metadata.get("file_path") == file_path:
                    nodes_to_delete.append(node_id)
            
            for node_id in nodes_to_delete:
                index.delete_ref_doc(node_id, delete_from_docstore=True)
            
            # 重新解析和添加文件
            if file_path.lower().endswith('.pdf'):
                if is_pdf_parseable(file_path):
                    text = extract_pdf_text(file_path)
                    if text.strip():
                        doc = Document(text=text, metadata=get_file_metadata(file_path))
                        index.insert(doc)
                        changes_made = True
                        print(f"✅ Updated PDF: {os.path.basename(file_path)}")
                    else:
                        print(f"⏭️ Skipped PDF (no text): {os.path.basename(file_path)}")
                else:
                    print(f"⏭️ Skipped PDF (not parseable): {os.path.basename(file_path)}")
            else:  # txt, md files
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                if text.strip():
                    doc = Document(text=text, metadata=get_file_metadata(file_path))
                    index.insert(doc)
                    changes_made = True
                    print(f"✅ Updated: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"⚠️ Error updating {file_path}: {e}")

    # 处理新增的文件
    if added_files:
        print(f"➕ Adding {len(added_files)} new files...")
        new_docs = []
        for file_path in added_files:
            if file_path.lower().endswith('.pdf'):
                print(f"🔍 Processing new PDF: {os.path.basename(file_path)}")
                if is_pdf_parseable(file_path):
                    text = extract_pdf_text(file_path)
                    if text.strip():
                        doc = Document(text=text, metadata=get_file_metadata(file_path))
                        new_docs.append(doc)
                        print(f"✅ Added PDF: {os.path.basename(file_path)}")
                    else:
                        print(f"⏭️ Skipped PDF (no text): {os.path.basename(file_path)}")
                else:
                    print(f"⏭️ Skipped PDF (not parseable): {os.path.basename(file_path)}")
            else:  # txt, md files
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    if text.strip():
                        doc = Document(text=text, metadata=get_file_metadata(file_path))
                        new_docs.append(doc)
                        print(f"✅ Added: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"⚠️ Error reading {file_path}: {e}")
        
        if new_docs:
            for doc in new_docs:
                index.insert(doc)
            changes_made = True
    
    if changes_made:
        print("💾 Persisting changes to the index...")
        index.storage_context.persist(persist_dir=storage_path)
        print("✨ Index synchronization complete.")

# --- 5. 主程序入口 ---

def main():
    """主函数，协调所有操作"""
    config = load_config()
    rag_config = config['rag']

    # 读取两个知识库路径和存储路径
    # 注意：这里的 'security_knowledge_base_path' 等键名需要与你的 conf.yaml 文件完全对应
    security_knowledge_path = rag_config.get('security_knowledge_docs_path', './security_knowledge')
    security_knowledge_storage = rag_config.get('security_knowledge_storage_path', './storage/security_knowledge')
    topsec_document_knowledge_path = rag_config.get('topsec_document_docs_path', './topsec_document_knowledge')
    topsec_document_knowledge_storage = rag_config.get('topsec_document_knowledge_storage_path', './storage/topsec_document_knowledge')

    setup_services()
    
    required_files = ["docstore.json", "vector_store.json", "index_store.json"]

    # 1. 处理 security_knowledge
    print("\n=== [1/2] 处理网络安全知识库 ===")
    is_complete_sec = all(os.path.exists(os.path.join(security_knowledge_storage, f)) for f in required_files)
    if not is_complete_sec:
        create_new_index(security_knowledge_path, security_knowledge_storage)
    else:
        synchronize_index(security_knowledge_path, security_knowledge_storage)

    # 2. 处理 topsec_document_knowledge
    print("\n=== [2/2] 处理天融信公司文档知识库 ===")
    is_complete_topsec = all(os.path.exists(os.path.join(topsec_document_knowledge_storage, f)) for f in required_files)
    if not is_complete_topsec:
        create_new_index(topsec_document_knowledge_path, topsec_document_knowledge_storage)
    else:
        synchronize_index(topsec_document_knowledge_path, topsec_document_knowledge_storage)

# --- 4. 调试功能 ---

def debug_index_content(storage_path: str, max_docs: int = 10):
    """调试功能：查看索引中的文档内容"""
    print(f"🔍 Debugging index content in: {storage_path}")
    
    if not os.path.exists(storage_path):
        print(f"❌ Storage path does not exist: {storage_path}")
        return
    
    try:
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        index = load_index_from_storage(storage_context)
        
        print(f"📊 Total documents in index: {len(index.docstore.docs)}")
        
        doc_count = 0
        for node_id, node in index.docstore.docs.items():
            if doc_count >= max_docs:
                break
                
            print(f"\n--- Document {doc_count + 1} ---")
            print(f"Node ID: {node_id}")
            
            if node.metadata:
                print("Metadata:")
                for key, value in node.metadata.items():
                    print(f"  {key}: {value}")
            else:
                print("Metadata: None")
            
            # 显示文档内容的前100个字符
            text_preview = node.text[:100] + "..." if len(node.text) > 100 else node.text
            print(f"Text preview: {text_preview}")
            
            doc_count += 1
        
        if len(index.docstore.docs) > max_docs:
            print(f"\n... and {len(index.docstore.docs) - max_docs} more documents")
            
    except Exception as e:
        print(f"❌ Error debugging index: {e}")

def clear_index(storage_path: str):
    """清空指定的索引（用于重建）"""
    print(f"🗑️ Clearing index in: {storage_path}")
    
    if os.path.exists(storage_path):
        import shutil
        shutil.rmtree(storage_path)
        print(f"✅ Index cleared: {storage_path}")
    else:
        print(f"⚠️ No index found at: {storage_path}")

if __name__ == "__main__":
    import sys
    
    # 检查是否有调试参数
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "debug":
            # 调试模式：显示索引内容
            setup_services()
            config = load_config()
            rag_config = config['rag']
            
            security_storage = rag_config.get('security_knowledge_storage_path', './storage/security_knowledge')
            topsec_storage = rag_config.get('topsec_document_knowledge_storage_path', './storage/topsec_document_knowledge')
            
            print("=== 网络安全知识库索引内容 ===")
            debug_index_content(security_storage)
            
            print("\n=== 天融信文档知识库索引内容 ===")
            debug_index_content(topsec_storage)
            
        elif command == "clear":
            # 清空索引模式
            config = load_config()
            rag_config = config['rag']
            
            security_storage = rag_config.get('security_knowledge_storage_path', './storage/security_knowledge')
            topsec_storage = rag_config.get('topsec_document_knowledge_storage_path', './storage/topsec_document_knowledge')
            
            if len(sys.argv) > 2:
                target = sys.argv[2]
                if target == "security":
                    clear_index(security_storage)
                elif target == "topsec":
                    clear_index(topsec_storage)
                elif target == "all":
                    clear_index(security_storage)
                    clear_index(topsec_storage)
                else:
                    print("用法: python upload_database.py clear [security|topsec|all]")
            else:
                print("用法: python upload_database.py clear [security|topsec|all]")
        else:
            print("可用命令:")
            print("  python upload_database.py         # 正常运行")
            print("  python upload_database.py debug   # 调试索引内容")
            print("  python upload_database.py clear [security|topsec|all]  # 清空索引")
    else:
        # 正常运行模式
        main()