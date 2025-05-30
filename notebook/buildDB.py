import os
from dotenv import load_dotenv, find_dotenv
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# 读取本地/项目的环境变量。
# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中  
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
def build_DB(user_file_path):
    _ = load_dotenv(find_dotenv())


    # 获取folder_path下所有文件路径，储存在file_paths里
    file_paths = []
    folder_path = user_file_path + '/knowledge_db'
    print(folder_path)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    print(file_paths)



    # 遍历文件路径并把实例化的loader存放在loaders里
    loaders = []

    for file_path in file_paths:
        file_type = file_path.split('.')[-1]
        if file_type == 'pdf':
            loaders.append(PyMuPDFLoader(file_path))
        elif file_type == 'md':
            loaders.append(UnstructuredMarkdownLoader(file_path))       
    # 下载文件并存储到text
    texts = []
    for loader in loaders: texts.extend(loader.load())

    # text = texts[1]
    # print(f"每一个元素的类型：{type(text)}.", 
    #     f"该文档的描述性数据：{text.metadata}", 
    #     f"查看该文档的内容:\n{text.page_content[0:]}", 
    #     sep="\n------\n")

    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)

    split_docs = text_splitter.split_documents(texts)




    embedding = ZhipuAIEmbeddings()



    persist_directory = user_file_path + '/vector_db/chroma'

    # !rm -rf '../../data_base/vector_db/chroma'  



    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory=persist_directory  # persist_directory目录保存到磁盘上
    )
    print(f"向量库中存储的数量：{vectordb._collection.count()}")
    
if __name__ == "__main__":
    build_DB('dxyu_DB')
    # build_DB('UserTMP')