import streamlit as st
from groq import Groq
import base64
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import tempfile

# Set up Groq API Key
os.environ['GROQ_API_KEY'] = `${{ secrets.API_KEY }}`

# Styling
canvas = st.markdown("""
    <style>
        header{ visibility: hidden; }   
    </style> """, unsafe_allow_html=True)


# Function to generate caption
def generate(uploaded_image, prompt):
    base64_image = base64.b64encode(uploaded_image.read()).decode('utf-8')
    client = Groq()
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{base64_image}',
                        },
                    },
                ],
            }
        ],
        model='meta-llama/llama-4-scout-17b-16e-instruct',
    )
    return chat_completion.choices[0].message.content

# Streamlit App
st.title("Spark AI")

tab_titles = [
    "Home",
    "Vision Instruct",
    "File Query",
]

tabs = st.tabs(tab_titles)

with tabs[0]:
    st.markdown("""
        <h4>Welcome to Spark AI!</h4>
        <p style="text-align: justify;">Unlock the power of AI-driven image and file analysis  our innovative application. Spark AI is designed to simplify complex tasks, providing accurate and efficient results. <a>SPARK AI</a></p>
        <h4>Advantages of the Spark AI</h4>
        <p style="text-align: justify;">It simplifies daily life tasks by using AI, generates the anlyzed data with in a minute. It saves the time by reading all data in files using AI-driven model.</p>
        <h4>Explore Our Features - Get Started</h4>
        <h5>Vision Instruct</h5>
        <p style="text-align: justify;">It is used to query with images. It let us analyze the image data by using the llama model.</p>
                """
                , unsafe_allow_html=True)

    st.markdown("""
        <h5>File Query</h5>
        <p style="text-align: justify;">It is used to query with files. It let us analyze the files like PDF, TXT and so on by using the llama model.</p>
    """, unsafe_allow_html=True)
    st.markdown("""
         <h4>About Spark AI</h4>
        <p style="text-indent: 60px; text-align: justify;"> Spark is an AI-powereed application developed as part of the Applied Artificial Intelligence: Practical Implementations course  by TechSaksham Program, which is a CSR initiative by Micrososft and SAP, implemented by Edunet Foundation</p>
        <br>""", unsafe_allow_html=True)    
    st.markdown("""
      <h4>Contact Us</h4>
        <p>For any queries or feedback, please reach out to us at <a>sathvikpalivela0@gmail.com</a>. 
  """, unsafe_allow_html=True)

with tabs[1]:
    #upload file
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
            # Show the uploaded image
            st.image(uploaded_file, caption='Uploaded Image')
            prompt = st.text_input('Enter Query')

            if st.button('Generate'):
                with st.spinner('Generating output...'):
                    if prompt:
                        output = generate(uploaded_file, prompt)
                    else:
                        output = generate(uploaded_file, 'What is in this picture?')
                st.subheader('Output:')
                st.write(output)

with tabs[2]:
    st.header("File Query")
    
    # Initialize LangChain components
    load_dotenv()
    
    # Initialize Groq
    groq_api_key = os.getenv('GROQ_API_KEY')
    llm = ChatGroq(api_key=groq_api_key, model_name="meta-llama/llama-4-scout-17b-16e-instruct")

    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
        Answer the question based on the provided context. 
        If you cannot find the answer in the context, say "I don't have enough information to answer this question."
        
        Context: {context}
        
        Question: {question}
        
        Answer: Let me help you with that.
    """)

    def create_vector_db(pdf_file):
        """Create a vector store from an uploaded PDF file"""
        if "vector_store" not in st.session_state:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_file.read())
                pdf_path = temp_file.name

            # Initialize embeddings
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name='all-MiniLM-L6-v2',
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e:
                st.error(f"Error initializing embeddings: {str(e)}")
                return

            # Load and process the PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            
            splits = text_splitter.split_documents(documents)
            
            # Create and store the vector store
            st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
            
            # Clean up temporary file
            os.unlink(pdf_path)

    # File upload section
    pdf_file = st.file_uploader("Upload a PDF file", type=['pdf'])

    if pdf_file:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                create_vector_db(pdf_file)
                st.success("PDF processed successfully! You can now ask questions about its content.")

    # Question answering section
    if "vector_store" in st.session_state:
        question = st.text_input("What would you like to know about the document?")
        
        if question and st.button("Ask"):
            with st.spinner("Finding answer..."):
                # Get relevant documents
                docs = st.session_state.vector_store.similarity_search(question)
                
                # Combine document content
                context = "\n\n".join(doc.page_content for doc in docs)
                
                # Create and run chain
                chain = prompt | llm | StrOutputParser()
                
                # Get response
                response = chain.invoke({
                    "context": context,
                    "question": question
                })
                
                # Display response
                st.write("### Answer")
                st.write(response)
