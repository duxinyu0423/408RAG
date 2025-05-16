import streamlit as st
# from langchain_openai import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
import sys
sys.path.append("notebook") # 将父目录放入系统路径中
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
from zhipuai_llm import ZhipuaiLLM
from langchain_core.prompts import MessagesPlaceholder
from langchain.retrievers import EnsembleRetriever
import os
import shutil
from buildDB import build_DB
def get_retriever_RAG():
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings()
    # 向量数据库持久化路径
    persist_directory = 'dxyu_DB/vector_db/chroma'
    print('检索本地向量数据库')
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectordb.as_retriever()

def get_retriever_RAG_temp():
    print('检索本地和临时向量数据库')
    persist_directories = ['dxyu_DB/vector_db/chroma', 'UserTMP/vector_db/chroma']
    embedding = ZhipuAIEmbeddings()
    retrievers = []
    
    for dir_path in persist_directories:
        vectordb = Chroma(
            persist_directory=dir_path,
            embedding_function=embedding
        )
        retrievers.append(vectordb.as_retriever())
    
    # 假设等权重合并，可调整 weights 参数
    ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=[1.0]*len(retrievers))
    return ensemble_retriever

def combine_docs_RAG(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])

def get_qa_history_chain_RAG():
    retriever = get_retriever_RAG()
    ## llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    llm = ZhipuaiLLM(model_name="glm-4-plus", temperature=0.1)
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
        "你是408助教小杜大王，可以为用户解决408相关的问题"
        # "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        # "如果用户问你你是谁，你就说你是杜欣瑀"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs_RAG)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieve_docs, 
        ).assign(answer=qa_chain)
    return qa_history_chain


# def get_retriever_RAG_temp():
#     embedding = ZhipuAIEmbeddings()
    
#     # 加载多个向量库
#     vectordb1 = Chroma(
#         persist_directory='dxyu_DB/vector_db/chroma',  # 第一个库路径
#         embedding_function=embedding
#     )
#     vectordb2 = Chroma(
#         persist_directory='UserTMP/vector_db/chroma',  # 第二个库路径
#         embedding_function=embedding
#     )
    
#     # 创建检索器并设置返回数量
#     retriever1 = vectordb1.as_retriever(search_kwargs={"k": 3})
#     retriever2 = vectordb2.as_retriever(search_kwargs={"k": 3})
    
#     # 合并检索器（假设权重相同）
#     ensemble_retriever = EnsembleRetriever(
#         retrievers=[retriever1, retriever2],
#         weights=[0.5, 0.5]  # 调整权重
#     )
#     return ensemble_retriever

def get_qa_history_chain_RAG_temp():
    retriever = get_retriever_RAG_temp()
    ## llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    llm = ZhipuaiLLM(model_name="glm-4-plus", temperature=0.1)
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
        "你是408助教小杜大王，可以为用户解决408相关的问题"
        # "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        # "如果用户问你你是谁，你就说你是杜欣瑀"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs_RAG)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieve_docs, 
        ).assign(answer=qa_chain)
    return qa_history_chain
            
def gen_response_RAG(chain, input, chat_history):
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]
            
def gen_response_LLM(chain, input, chat_history):
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        yield res
            

def get_qa_history_chain_LLM():
    # 初始化纯LLM（此处保留原参数配置）
    llm = ZhipuaiLLM(model_name="glm-4-plus", temperature=0.1)
    
    # 精简后的系统提示（移除检索相关指令）
    system_prompt = (
        "请用简洁易懂的语言回答用户提问。"
        "若不知道答案请说不知道。"
        "请使用简洁的话语回答用户。"
    )
    
    # 构建纯对话提示模板
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),  # 保留对话历史占位
        ("human", "{input}"),
    ])

    # 简化处理链（移除所有检索相关操作）
    qa_chain = (
        RunnablePassthrough()  # 直接传递输入
        | qa_prompt 
        | llm
        | StrOutputParser()
    )

    # 最终对话链（保留多轮对话能力）
    return qa_chain



            
            
from tempfile import TemporaryDirectory
import os            


# Streamlit 应用程序界面
def main():
    st.markdown('### 🦜🔗 小杜大王教你408')

    # 用于跟踪对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.toast('👋(≧∇≦)ﾉ 我是小杜大王，有什么问题尽管问我吧！✨ 小提示：可以上传你自己的文件哦~💡 试试问我："TCP和UDP有什么区别？"')
        
        
    if "use_rag" not in st.session_state:  # 新增：控制RAG开关
        st.session_state.use_rag = True
        
        # 存储检索问答链
    if "qa_history_chain" not in st.session_state and st.session_state.use_rag:
        st.session_state.qa_history_chain = get_qa_history_chain_RAG()
    elif "qa_history_chain" not in st.session_state and not st.session_state.use_rag:
        st.session_state.qa_history_chain = get_qa_history_chain_LLM()

    # 添加三个功能按钮（横向排列）
    col1, col2, col3 = st.columns(3)
    
    

    with col1:
        # 文件上传按钮
        uploaded_file = st.file_uploader("📁 添加文件到RAG", 
                                    type=["pdf", "txt", "md"],
                                    accept_multiple_files=False,
                                    key="file_uploader")
        last_folder_path = 'UserTMP/knowledge_db'
        last_name = ''
        if os.path.isdir(last_folder_path):
            # 如果存在这个路径
            # 获取该路径下文件名字
            file_names = os.listdir(last_folder_path)
            if len(file_names) == 1:
                
                last_name = file_names[0]
        if uploaded_file:
            if last_name == uploaded_file.name:
                print("🚫 请不要上传重复的文件！")
            else:
                print("🎉 文件上传成功！")
                # global last_name
                print('文件名：', uploaded_file.name)
                # folder_path = '/home/user/lwh/dxyu/llm-universe-test/UserTMP'
                folder_path = 'UserTMP'
                # 检查路径是否为目录
                if os.path.isdir(folder_path):
                    # 递归删除目录及其内容
                    shutil.rmtree(folder_path)
                    print(f"文件夹 '{folder_path}' 已删除。")
                else:
                    print(f"文件夹 '{folder_path}' 不存在。")
                os.mkdir(folder_path)
                target_folder = folder_path + '/knowledge_db'
                os.mkdir(folder_path + '/knowledge_db')
                print(f"文件夹 '{folder_path}' 创建成功。")  
                
                save_path = os.path.join(target_folder, uploaded_file.name)          
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                print('开始创建临时数据库')
                build_DB(folder_path)
                print('临时数据库创建成功')
                st.session_state.qa_history_chain = get_qa_history_chain_RAG_temp()  # 重新加载链
                st.toast("文件已成功添加到知识库！", icon="✅")
                print('col1')

    

    with col2:
        # 纯LLM模式按钮
        if st.button("🤖 仅LLM模式", 
                    help="直接使用大模型生成回答",
                    type="primary" if not st.session_state.use_rag else "secondary"):
            st.toast('👋(≧∇≦)ﾉ 我是小杜大王，有什么问题尽管问我吧！\n ✨ 小提示：可以上传你自己的文件哦~\n💡 试试问我："TCP和UDP有什么区别？"')
            st.session_state.use_rag = False
            st.session_state.qa_history_chain = get_qa_history_chain_LLM()  # 重新加载链
            st.rerun()
            print('col2')

    with col3:
        # RAG模式按钮
        if st.button("🔍 RAG模式", 
                    help="结合知识库检索生成回答",
                    type="primary" if st.session_state.use_rag else "secondary"):
            st.session_state.use_rag = True
            st.session_state.qa_history_chain = get_qa_history_chain_RAG()  # 重新加载链
            st.rerun()
            print('col3')

            
            
    messages = st.container(height=450)
    # 显示整个对话历史
    for message in st.session_state.messages:
            with messages.chat_message(message[0]):
                st.write(message[1])
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        with messages.chat_message("human"):
            st.write(prompt)
        answer = ""
        if st.session_state.use_rag:
            answer = gen_response_RAG(
                chain=st.session_state.qa_history_chain,
                input=prompt,
                chat_history=st.session_state.messages
            )
        else:
            answer = gen_response_LLM(
                chain=st.session_state.qa_history_chain,
                input=prompt,
                chat_history=st.session_state.messages
            )
        with messages.chat_message("ai"):
            output = st.write_stream(answer)
        st.session_state.messages.append(("ai", output))
    


if __name__ == "__main__":
    main()
    
