import streamlit as st
import pandas as pd
import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from llm_config import get_model
import os
import openai

# Using a consistent embedding model
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Providing title of the app
st.title("🔍 DevOps & Developer Assistance")

# Role selection
role = st.selectbox("Choose your role", ["Developer", "DevOps Engineer", "Product Owner"])

# Function to load csv dataset
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # Remove extra spaces in column names
    df['Error Message'] = df[['Error Type', 'Description', 'Possible Causes', 'Solution']].astype(str).agg(' | '.join,
                                                                                                           axis=1)
    return df


# Function to create Normalized Embeddings
def create_embeddings(df):
    vectors = np.array(embedding_model.embed_documents(df['Error Message'].tolist()), dtype='float32')
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # Normalize embeddings
    return vectors


# Function to store embeddings in FAISS database using cosine similarity
def store_embeddings_faiss(df, vectors, faiss_index_path):
    vector_dimension = vectors.shape[1]  # Getting the dimension of vector(embedding)
    index = faiss.IndexFlatIP(vector_dimension)  # Using Inner Product for Cosine Similarity
    index.add(vectors)
    faiss.write_index(index, faiss_index_path)

    documents = [Document(page_content=text) for text in df['Error Message'].tolist()]
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    vectorstore = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)}),
        index_to_docstore_id=index_to_docstore_id
    )
    return vectorstore


# Step 4: Load FAISS Index with Dimension Check
def load_faiss_index(faiss_index_path, df):
    if os.path.exists(faiss_index_path):
        index = faiss.read_index(faiss_index_path)
        expected_dim = create_embeddings(df).shape[1]
        if index.d != expected_dim:
            st.warning(f"⚠️ FAISS index dimension mismatch! Rebuilding index...")
            os.remove(faiss_index_path)
            return None  # Force recreation
        return index
    return None


# Step 5: Initialize LLM
def initialize_llm():
    llm = get_model()
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=""" 
        You are a DevOps assistant. Use the provided context to help resolve user errors.

        Context: {context}

        User Question: {question}

        Provide a clear and concise solution.
        """
    )
    return llm, prompt_template


# Step 6: Setup Retrieval-based QA System
def setup_retrieval_qa(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=initialize_llm()[0],
        retriever=retriever,
        chain_type="stuff",
    )
    return qa_chain


# Load Data and Setup FAISS
csv_path = "E:\AI_DevOps 1\sample.csv"
faiss_index_path = "faiss_index.bin"
df = load_data(csv_path)

index = load_faiss_index(faiss_index_path, df)
if index is None:
    vectors = create_embeddings(df)
    vectorstore = store_embeddings_faiss(df, vectors, faiss_index_path)
else:
    documents = [Document(page_content=text) for text in df['Error Message'].tolist()]
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}
    vectorstore = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)}),
        index_to_docstore_id=index_to_docstore_id
    )

qa_chain = setup_retrieval_qa(vectorstore)

# Function to analyze code
def analyze_code(code):
    try:
        # Basic check to see if the code is valid (simple syntax check for Python code)
        compiled_code = compile(code, "<string>", "exec")

        # If no error occurs, capture output
        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        exec(compiled_code)

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        return f"Success: The code has been successfully executed. Output: {output}"

    except Exception as e:
        # If an error occurs, display failure and print the error message
        return f"Error: {str(e)}"


# Function to resolve error
def resolve_error(error_message):
    # Assuming the `qa_chain.run` is from the error resolution logic set up previously
    response = qa_chain.run(error_message)
    return response


# Streamlit UI for user input based on role
if role == "Developer":
    st.subheader("🔧 Analyze Your Code")
    code_input = st.text_area("Paste your code here", height=300)
    if st.button("Analyze Code"):
        if code_input:
            # Call the function to analyze the code (you can modify `analyze_code` logic as required)
            response = analyze_code(code_input)
            st.write("### Result:")
            st.write(response)
        else:
            st.error("Please provide code to analyze.")


elif role == "DevOps Engineer":
    st.subheader("🔧 DevOps Engineer Tools")

    # Create a radio button to switch between "Resolve Error" and "Generate Code"
    tool_option = st.radio("Choose a tool:", ["Resolve Error", "Generate Code"])

    if tool_option == "Resolve Error":
        st.subheader("🔧 Resolve Error")

        # Input box for entering error message
        user_question = st.text_area("Enter your error message:", height=150)

        # Button to process the error message
        if st.button("Resolve Error") and user_question:
            # Check if the user is asking for code
            if not any(word in user_question.lower() for word in ["code", "script", "snippet", "example"]):
                user_question += " Give steps to solve this."

            query_vector = embedding_model.embed_query(user_question)
            query_vector /= np.linalg.norm(query_vector)  # Normalize query vector

            if len(query_vector) == vectorstore.index.d:
                response = qa_chain.run(user_question)
                st.subheader("💡 Suggested Solution:")
                st.write(response)
            else:
                st.error(f"⚠️ Query dimension ({len(query_vector)}) does not match FAISS index ({vectorstore.index.d}).")
        elif user_question == "":
            st.error("Please enter an error message to resolve.")

    elif tool_option == "Generate Code":
        st.subheader("💻 Generate Code")

        # Input box for entering code generation request
        code_request = st.text_area("Describe the code you need:", height=150)

        # Button to generate code
        if st.button("Generate Code") and code_request:
            # Modify the prompt to focus on generating code
            code_prompt_template = """
            You are a DevOps Engineer. Generate a code snippet or script based on the following request:
            Request: {code_request}
            Provide the code in a clear and executable format.
            """
            code_prompt = code_prompt_template.format(code_request=code_request)

            # Use the LLM to generate the code
            response = qa_chain.run(code_prompt)
            st.subheader("💻 Generated Code:")
            st.code(response, language="python")  # Display the code in a code block
        elif code_request == "":
            st.error("Please describe the code you need to generate.")



elif role == "Product Owner":
    st.subheader("📊 View Insights")
    st.write("As a Product Owner, you can track issues and collaborate with your development and DevOps teams.")
    st.write("For more details on project management and progress tracking, please contact the relevant team.")

# Function to analyze code (same as before, or modify it as necessary)
def analyze_code(code):
    prompt_template = """
    You are an AI assistant helping developers. Analyze the provided code and explain any issues, improvements, or suggestions for better performance and readability.
    Code: {code}
    """
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_template.format(code=code),
        temperature=0.5,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Function to resolve error (same as before)
def resolve_error(error_message):
    # Assuming the `qa_chain.run` is from the error resolution logic set up previously
    response = qa_chain.run(error_message)
    return response