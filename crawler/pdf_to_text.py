import os
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PDFProcessor:
    def __init__(self, tesseract_cmd=None):
        """
        初始化PDF处理器
        :param tesseract_cmd: Tesseract可执行文件的路径
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
    def extract_text_from_pdf(self, pdf_path, output_txt_path):
        """
        从PDF中提取文本，包括OCR识别图片中的文字
        :param pdf_path: PDF文件路径
        :param output_txt_path: 输出文本文件路径
        """
        try:
            os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

            with pdfplumber.open(pdf_path) as pdf:
                all_text = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    logging.info(f"正在处理第 {page_num} 页...")
                    
                    # 尝试直接提取文本
                    text = page.extract_text()
                    
                    # 没有文本，对图片进行OCR
                    if not text or len(text.strip()) < 10:
                        logging.info(f"第 {page_num} 页可能是图片，进行OCR识别...")

                        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
                        
                        for img in images:
                            text = pytesseract.image_to_string(img, lang='chi_sim+eng')
                    
                    if text:
                        all_text.append(f"\n=== 第 {page_num} 页 ===\n")
                        all_text.append(text)
            
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(all_text))
            
            logging.info(f"文本提取完成，已保存到: {output_txt_path}")
            return True
            
        except Exception as e:
            logging.error(f"处理PDF时发生错误: {str(e)}")
            return False

def main():
    # 设置Tesseract路径
    tesseract_path = r"D:\Program Files\Tesseract-OCR\tesseract.exe" 
    
    processor = PDFProcessor(tesseract_cmd=tesseract_path)
    
    input_dir = "./knowledge_base" 
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    print(f"找到 {len(pdf_files)} 个PDF文件，开始处理...")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        pdf_filename = os.path.splitext(pdf_file)[0]
        output_path = os.path.join(output_dir, f"{pdf_filename}.txt")
        
        print(f"\n正在处理: {pdf_file}")
        success = processor.extract_text_from_pdf(pdf_path, output_path)
        
        if success:
            print(f"处理成功！输出文件保存在：{output_path}")
        else:
            print(f"处理失败：{pdf_file}")
    
    print("\n所有文件处理完成！")

if __name__ == "__main__":
    main() 