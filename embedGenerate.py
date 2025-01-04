import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
import boto3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from flashrank import Ranker
from langchain.retrievers.document_compressors import FlashrankRerank

# Load environment variables
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("aws_access_key_id")
AWS_SECRET_ACCESS_KEY = os.getenv("aws_secret_access_key")
AWS_TEST_OUTPUT_BUCKET = os.getenv("aws_test_output_bucket")
GOOGLE_API_KEY = os.getenv("API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Global variables
embeddings = None
vectorstore = None
PERSIST_DIRECTORY = "chroma_db"

def initialize_embeddings(file_path):
    """Initialize embeddings if they haven't been created yet"""
    global embeddings, vectorstore
    
    # Check if vectorstore already exists
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        print("Loading existing embeddings...")
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        return True
        
    try:
        print("Creating new embeddings...")
        embeddings = OpenAIEmbeddings()
        loader = CSVLoader(file_path=file_path, encoding='utf-8')
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
        final_documents = text_splitter.split_documents(docs)
        
        for id, final_document in enumerate(final_documents):
            final_document.metadata["id"] = id
            
        vectorstore = Chroma.from_documents(
            final_documents, 
            embeddings, 
            persist_directory=PERSIST_DIRECTORY
        )
        print("Embedding completed, and vector database is ready.")
        return True
        
    except Exception as e:
        print(f"An error occurred during embedding: {e}")
        import traceback
        traceback.print_exc()
        return False

async def generating_defect(issues):
    global vectorstore
    
    # Initialize retrieval components
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
    model_name = "ms-marco-TinyBERT-L-2-v2"
    flashrank_client = Ranker(model_name=model_name)
    compressor = FlashrankRerank(client=flashrank_client, top_n=3, model=model_name)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
    )

    consolidated_responses = []
    
    prompt = ChatPromptTemplate.from_template("""
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
    """)

    for issue in issues:
        compressed_docs = compression_retriever.invoke(issue)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(compression_retriever, document_chain)
        responses = retrieval_chain.invoke({'input': issue})
        consolidated_responses.append(responses['answer'])

    consolidated_csv = "\n".join(consolidated_responses)
    
    # Save to S3
    response = s3_client.put_object(
        Bucket=AWS_TEST_OUTPUT_BUCKET,
        Key="TestDefect1.csv",
        Body=consolidated_csv
    )

    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    url = f"https://{AWS_TEST_OUTPUT_BUCKET}.s3.amazonaws.com/TestDefect1.csv"
    print(url)
    
    return url if status == 200 else None

async def handle_defect_detection_button_click(filepath, issues):
    """Combined handler for both embedding and defect detection"""
    global vectorstore
    
    # Only initialize embeddings if they don't exist
    if vectorstore is None:
        success = initialize_embeddings(filepath)
        if not success:
            return "Error: Failed to initialize embeddings"
    
    # Process the defects
    try:
        result_url = await generating_defect(issues)
        return result_url if result_url else "Error: Failed to generate defects report"
    except Exception as e:
        print(f"Error in defect detection: {e}")
        return f"Error: {str(e)}"