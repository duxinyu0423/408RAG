# 408助教小杜

## 一、项目简介

本项目使用智谱清言API+RAG技术的智能问答系统。本系统通过对408资料建立向量数据库和检索文档来提升对408相关问题回答的准确性。

## 二、部署说明

### 1.初始化项目

```bash
uv init llm-universe-test
```

### 2.创建环境

```bash
uv venv llm-universe-test --python 3.13
```

### 3.激活环境

```bash
source llm-universe-test/bin/activate
```

### 4.安装re依赖

- 直接装

  ```bash
  uv pip install -r requirements.txt
  ```

- 清华源

  ```bash
  uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```


### 5.升级包

```bash
uv pip install --upgrade zhipuai langchain
```

### 6.包降级

```bash
uv add pydantic==2.9.2
```

### 7.设置API

打开.env文件，填入API

```bash
ZHIPUAI_API_KEY="自己的质谱API"
```

### 8.运行代码

```bash
streamlit run "notebook/streamlit_app.py"
```
