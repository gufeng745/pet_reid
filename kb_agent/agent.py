"""
知识库 Agent - 基于文档的问答系统
支持阿里云 Qwen API - 使用 TF-IDF 检索（无需外部嵌入 API）
"""
import os
import hashlib
import json
from typing import List, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class TextLoader:
    """加载文本文件"""

    def __init__(self, directory: str):
        self.directory = directory
        self.supported_extensions = ['.txt', '.md', '.pdf', '.docx']

    def load_all(self) -> List[dict]:
        """加载目录下所有支持的文本文件"""
        documents = []

        for filename in os.listdir(self.directory):
            filepath = os.path.join(self.directory, filename)

            if not os.path.isfile(filepath):
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext == '.txt' or ext == '.md':
                content = self._load_text(filepath)
                documents.append({
                    'content': content,
                    'source': filename,
                    'metadata': {'filename': filename}
                })
            elif ext == '.pdf':
                content = self._load_pdf(filepath)
                if content:
                    documents.append({
                        'content': content,
                        'source': filename,
                        'metadata': {'filename': filename}
                    })
            elif ext == '.doc' or ext == '.docx':
                content = self._load_word(filepath)
                if content:
                    documents.append({
                        'content': content,
                        'source': filename,
                        'metadata': {'filename': filename}
                    })

        print(f"已加载 {len(documents)} 个文档")
        return documents

    def _load_text(self, filepath: str) -> str:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_pdf(self, filepath: str) -> Optional[str]:
        try:
            import pypdf
            reader = pypdf.PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except ImportError:
            print("需要安装 pypdf: pip install pypdf")
            return None

    def _load_word(self, filepath: str) -> Optional[str]:
        """加载 Word 文档"""
        try:
            ext = os.path.splitext(filepath)[1].lower()
            if ext == '.doc':
                # .doc 文件使用 pywin32 处理（仅 Windows）
                try:
                    import win32com.client
                    # 静默启动 Word 应用
                    word = win32com.client.Dispatch("Word.Application")
                    word.Visible = False
                    word.DisplayAlerts = False

                    abs_path = os.path.abspath(filepath)
                    doc = word.Documents.Open(abs_path)

                    # 使用 Range().Text 获取更清晰的内容
                    # 尝试获取全文内容
                    if doc.Content and doc.Content.Text:
                        text = doc.Content.Text
                    else:
                        # 尝试逐段读取
                        text = ""
                        for para in doc.Paragraphs:
                            if para.Range.Text:
                                text += para.Range.Text + "\n"

                    doc.Close(False)
                    word.Quit()

                    # 清理文本
                    text = self._clean_word_text(text)
                    return text
                except ImportError:
                    print(".doc 文件需要 pywin32: pip install pywin32 (仅 Windows)")
                    return None
                except Exception as e:
                    print(f"使用 pywin32 加载 .doc 文件失败：{e}")
                    return None
            elif ext == '.docx':
                from docx import Document
                doc = Document(filepath)
                text = "\n\n".join([para.text for para in doc.paragraphs if para.text])
                return self._clean_word_text(text)
        except Exception as e:
            print(f"加载 Word 文档失败：{e}")
            return None

    def _clean_word_text(self, text: str) -> str:
        """清理 Word 文档文本"""
        import re
        # 移除多余的控制字符和多余空白
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        # 移除多余换行
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # 移除首尾空白
        text = text.strip()
        return text


class TextSplitter:
    """文本分块"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        """将文本分割成块"""
        chunks = []

        # 先按段落分割
        paragraphs = text.split('\n\n')

        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果当前段落 + 已有内容超过限制，保存当前块并开始新的
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                # 如果单个段落就超长，进一步分割
                if len(para) > self.chunk_size:
                    sentences = para.replace('.', '。\n').replace('!', '！\n').replace('?', '？\n').split('\n')
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) > self.chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence
                        else:
                            current_chunk += sentence
                else:
                    current_chunk = para
            else:
                current_chunk += para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def split_documents(self, documents: List[dict]) -> List[dict]:
        """对多个文档进行分块"""
        all_chunks = []

        for doc in documents:
            chunks = self.split(doc['content'])
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'content': chunk,
                    'source': doc['source'],
                    'chunk_id': i,
                    'metadata': {**doc['metadata'], 'chunk_id': i}
                })

        print(f"分割成 {len(all_chunks)} 个文本块")
        return all_chunks


class TFIDFRetriever:
    """使用 TF-IDF 进行文本检索"""

    def __init__(self):
        self.chunks = []
        self.vocabulary = {}
        self.idf = {}
        self.tfidf_vectors = []

    def _tokenize(self, text: str) -> List[str]:
        """简单的中文分词（按字符和常用词）"""
        # 简单处理：移除标点，返回字符列表
        import re
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        return list(text)

    def _compute_tf(self, text: str) -> dict:
        """计算词频"""
        tokens = self._tokenize(text)
        tf = {}
        total = len(tokens)
        if total == 0:
            return tf
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        for token in tf:
            tf[token] /= total
        return tf

    def fit(self, chunks: List[dict]):
        """拟合一 TF-IDF 模型"""
        self.chunks = chunks
        n_docs = len(chunks)

        # 构建词汇表和文档频率
        doc_freq = {}
        for chunk in chunks:
            tokens = set(self._tokenize(chunk['content']))
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)

        # 计算 IDF
        import math
        self.idf = {}
        for token, df in doc_freq.items():
            self.idf[token] = math.log((n_docs + 1) / (df + 1)) + 1

        # 计算每个文档的 TF-IDF 向量
        self.tfidf_vectors = []
        for chunk in chunks:
            tf = self._compute_tf(chunk['content'])
            vector = {}
            for token, tf_val in tf.items():
                if token in self.idf:
                    vector[token] = tf_val * self.idf[token]
            self.tfidf_vectors.append(vector)

        print(f"TF-IDF 索引构建完成，词汇表大小：{len(self.vocabulary)}")

    def _cosine_similarity(self, vec1: dict, vec2: dict) -> float:
        """计算余弦相似度"""
        import math

        # 找到共同的 token
        common_tokens = set(vec1.keys()) & set(vec2.keys())
        if not common_tokens:
            return 0.0

        # 计算点积
        dot_product = sum(vec1[t] * vec2[t] for t in common_tokens)

        # 计算模长
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def query(self, query_text: str, n_results: int = 3) -> dict:
        """检索相关文档"""
        import math

        # 计算查询的 TF-IDF 向量
        query_tf = self._compute_tf(query_text)
        query_vector = {}
        for token, tf_val in query_tf.items():
            if token in self.idf:
                query_vector[token] = tf_val * self.idf[token]

        # 计算与所有文档的相似度
        similarities = []
        for i, doc_vector in enumerate(self.tfidf_vectors):
            sim = self._cosine_similarity(query_vector, doc_vector)
            similarities.append((i, sim))

        # 排序并返回 Top-N
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:n_results]]

        results = {
            'documents': [[self.chunks[i]['content'] for i in top_indices]],
            'metadatas': [[self.chunks[i]['metadata'] for i in top_indices]],
            'distances': [[1 - sim for _, sim in similarities[:n_results]]]
        }

        return results


class KBAgent:
    """知识库问答 Agent"""

    def __init__(
        self,
        knowledge_base_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.knowledge_base_path = knowledge_base_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化组件
        self.loader = TextLoader(knowledge_base_path)
        self.splitter = TextSplitter(chunk_size, chunk_overlap)
        self.retriever = TFIDFRetriever()

        # 初始化 API
        self._init_api()

    def _init_api(self):
        """初始化 API"""
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.api_base = os.getenv('OPENAI_API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1')

        if not self.api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")

    def build_index(self):
        """构建知识索引"""
        print("正在加载文档...")
        documents = self.loader.load_all()

        print("正在分割文本...")
        chunks = self.splitter.split_documents(documents)

        print("正在构建 TF-IDF 索引...")
        self.retriever.fit(chunks)

        print("索引构建完成!")

    def query(self, question: str, n_results: int = 3) -> str:
        """回答问题"""
        from openai import OpenAI

        # 检索相关文档
        results = self.retriever.query(question, n_results)

        # 构建上下文
        context_parts = []

        for i, doc in enumerate(results['documents'][0]):
            context_parts.append(f"[片段{i + 1}]: {doc}")

        context = "\n\n".join(context_parts)

        # 构建提示
        system_prompt = "你是一个知识库助手，基于提供的参考资料回答问题。如果资料中没有相关信息，请如实说明。"
        user_prompt = f"""参考资料：
{context}

问题：{question}

请用中文回答："""

        # 创建 OpenAI 客户端（使用阿里云兼容端点）
        api_key = os.getenv('OPENAI_API_KEY')
        api_base = os.getenv('OPENAI_API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        model = os.getenv('OPENAI_MODEL', 'qwen3.5-plus')

        client = OpenAI(api_key=api_key, base_url=api_base)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1024
        )

        return response.choices[0].message.content

    def chat_loop(self):
        """交互式问答循环"""
        import sys

        print("\n" + "=" * 50)
        print("知识库问答系统 - 按 Ctrl+C 或输入 'quit' 退出")
        print("=" * 50 + "\n")

        # 检查是否是交互模式
        if not sys.stdin.isatty():
            print("警告：非交互模式，无法接收输入")
            return

        while True:
            try:
                question = input("\n你：").strip()

                if not question:
                    continue
                if question.lower() in ['quit', 'exit', '退出']:
                    print("再见!")
                    break

                print("\n思考中...", end=" ")
                answer = self.query(question)
                print(f"\n\n助手：{answer}")

            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except EOFError:
                print("\n\n输入结束，退出!")
                break


def main():
    # 配置
    KNOWLEDGE_BASE = "./knowledge_base"

    # 创建知识库目录（如果不存在）
    os.makedirs(KNOWLEDGE_BASE, exist_ok=True)
    print(f"知识库目录：{KNOWLEDGE_BASE}")

    # 列出知识库中的文件
    if os.path.exists(KNOWLEDGE_BASE):
        files = os.listdir(KNOWLEDGE_BASE)
        print(f"知识库中的文件：{files}")

    # 创建 Agent
    agent = KBAgent(
        knowledge_base_path=KNOWLEDGE_BASE,
        chunk_size=500,
        chunk_overlap=50
    )

    # 构建索引
    agent.build_index()

    # 测试问题（根据 law.doc 法律文档定制）
    test_questions = [
        "案件的原告和被告分别是谁？",
        "案件的基本事实是什么？",
        "法院的判决结果是什么？"
    ]

    print("\n" + "=" * 50)
    print("测试问答")
    print("=" * 50 + "\n")

    for question in test_questions:
        print(f"问题：{question}")
        print("思考中...", end=" ")
        answer = agent.query(question)
        print(f"\n回答：{answer}\n")
        print("-" * 50)

    # 开始交互问答
    agent.chat_loop()


if __name__ == "__main__":
    main()
