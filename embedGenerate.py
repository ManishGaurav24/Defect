import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
import time
import boto3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import asyncio
from langchain.retrievers import ContextualCompressionRetriever
from flashrank import Ranker
from langchain.retrievers.document_compressors import FlashrankRerank

# Load environment variables
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("aws_access_key_id")
AWS_SECRET_ACCESS_KEY = os.getenv("aws_secret_access_key")
AWS_TEST_OUTPUT_BUCKET = os.getenv("aws_test_output_bucket")
# Set OPENAI_API_KEY as an environment variable
GOOGLE_API_KEY = os.getenv("API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    # aws_session_token=AWS_SESSION_TOKEN,
)

# Global variables to hold embeddings and vector store
embeddings = None
vectorstore = None

def vector_embedding(file_path):
    global embeddings, vectorstore

    try:
        embeddings = OpenAIEmbeddings()
        loader = CSVLoader(file_path=file_path, encoding='utf-8')
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
        final_documents = text_splitter.split_documents(docs)
        for id, final_document in enumerate(final_documents):
            final_document.metadata["id"] = id
        vectorstore = Chroma.from_documents(final_documents, embeddings, persist_directory="chroma_db")
        # vectorstore.persist()
        if vectorstore:
            print("Embedding completed, and vector database is ready.")
    except Exception as e:
        print(f"An error occurred during embedding: {e}")
        import traceback
        traceback.print_exc()
    
async def generating_defect(issues):
    global vectorstore
    if vectorstore is None:
        vectorstore = Chroma(persist_directory="chroma_db", embedding_function=OpenAIEmbeddings())
        if vectorstore is None:
            raise ValueError("Vectorstore not initialized. Please run embeddings first.")

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
    
    flashrank_client = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    compressor = FlashrankRerank(client=flashrank_client, top_n=3)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=vectorstore.as_retriever())

    # Consolidated responses
    consolidated_responses = []

    for issue in issues:  # Iterate through the array of issues
        compressed_docs = compression_retriever.invoke(issue)
        print("Start\n")
        print(compressed_docs)
        print("\nEnd")

        prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question
            <context>
            {context}
            <context>
            Question: Given the following context, find all issues that have the same meaning as this {input}.
            For each issue, check if the relevance score is greater than 0.6.
            Return the Issue Key, Summary, Project name, Assignee, Components and input. 
            Output the results in CSV format with the following columns: Input, Issue Key, Summary,Project name, Assignee and Components.
            If no similar defects are found, return the Input along with "Not Found" in all other fields.
            """
        )
        
        # in which issue contains id which is a number and actual issue seperated by colon. You have to conside the actual issue only.
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(compression_retriever, document_chain)

        responses = retrieval_chain.invoke({'input': issue})
        consolidated_responses.append(responses['answer'])  # Collect responses for all issues

    # Consolidate the responses into one CSV output
    consolidated_csv = "\n".join(consolidated_responses)

    # Save the consolidated CSV to S3
    response = s3_client.put_object(
        Bucket=AWS_TEST_OUTPUT_BUCKET, Key=f"TestDefect1.csv", Body=consolidated_csv
    )

    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    url = f"https://{AWS_TEST_OUTPUT_BUCKET}.s3.amazonaws.com/TestDefect1.csv"
    print(url)

    if status == 200:
        return url
    else:
        return None

def handle_start_embedding_button_click(filepath):
    print("Start Embedding button clicked...")
    vector_embedding(filepath)

def handle_defect_detection_button_click(issue):
    print("Start Generating defect...")
    return asyncio.run(generating_defect(issue))
