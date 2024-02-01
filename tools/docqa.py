from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.schema import Document
import streamlit as st
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from llm import llm, embeddings
from langchain.chains import RetrievalQA

# A list of Documents
documents = [
    Document(
        page_content="The Star Wars franchise depicts the adventures of characters A long time ago in a galaxy far, far away across multiple fictional eras, in which humans and many species of aliens (often humanoid) co-exist with robots (typically referred to in the films as 'droids'), which may be programmed for personal assistance or battle. Space travel between planets is common due to lightspeed hyperspace technology. The planets range from wealthy, planet-wide cities to deserts scarcely populated by primitive tribes. Virtually any Earth biome, along with many fictional ones, has its counterpart as a Star Wars planet which, in most cases, teem with sentient and non-sentient alien life. The franchise also makes use of other astronomical objects such as asteroid fields and nebulae. Spacecraft range from small starfighters to large capital ships, such as the Star Destroyers, as well as space stations such as the moon-sized Death Stars. Telecommunication includes two-way audio and audiovisual screens, holographic projections and hyperspace transmission.",
        metadata={"source": "local"}
    )
]

# Service used to create the embeddings
embedding_provider = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

new_vector = Neo4jVector.from_documents(
    documents,
    embedding_provider,
    url=st.secrets["NEO4J_URI"],             # (2)
    username=st.secrets["NEO4J_USERNAME"],   # (3)
    password=st.secrets["NEO4J_PASSWORD"],   # (4)
    index_name="myVectorIndex",
    node_label="Chunk",
    text_node_property="text",
    embedding_node_property="embedding",
    create_id_index=True,
)


retriever = new_vector.as_retriever()

doc_qa = RetrievalQA.from_chain_type(
    llm,                  # (1)
    chain_type="stuff",   # (2)
    retriever=retriever,  # (3)
)

# tag::generate-response[]
def generate_response(prompt):
    """
    Use the Neo4j Vector Search Index
    to augment the response from the LLM
    """

    # Handle the response
    response = doc_qa({"question": prompt})

    return response['answer']
# end::generate-response[]