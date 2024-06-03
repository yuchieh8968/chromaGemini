import chromadb
import pathlib
import polars as pl
from more_itertools import batched
from chromadb.utils import embedding_functions
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import time, os
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv()) # read local .env file

# safety settings for Gemini
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_ONLY_HIGH",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_ONLY_HIGH",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_ONLY_HIGH",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_ONLY_HIGH",
  },
]

# config for gemini
generation_config = {
  "temperature": 0.6, #Randomness: 0 unrandom, 1 random
  "top_p": 1,# samples tokens with the highest probability scores until the sum of the scores reaches the specified threshold value. 
  "top_k": 50, # samples tokens with the highest probabilities until the specified number of tokens is reached.
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

#specify which gemini model to use
model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  safety_settings=safety_settings,
  generation_config=generation_config,
)

# setup API key
genai.configure(api_key=os.environ['GEMINI_KEY'])

# specify paths and variables
DATA_PATH = "./archive/*"
CHROMA_PATH = "./car_review_embeddings"
EMBEDDING_FUNC_NAME = "multi-qa-MiniLM-L6-cos-v1"
COLLECTION_NAME = "car_reviews"

def prepare_car_reviews_data(data_path: pathlib.Path, vehicle_years: list[int] = [2017]):
    """Prepare the car reviews dataset for ChromaDB"""

    # Define the schema to ensure proper data types are enforced
    dtypes = {
        "": pl.Int64,
        "Review_Date": pl.Utf8,
        "Author_Name": pl.Utf8,
        "Vehicle_Title": pl.Utf8,
        "Review_Title": pl.Utf8,
        "Review": pl.Utf8,
        "Rating": pl.Float64,
    }

    # Scan the car reviews dataset(s)
    car_reviews = pl.scan_csv(data_path, dtypes=dtypes)

    # Extract the vehicle title and year as new columns
    # Filter on selected years
    car_review_db_data = (
        car_reviews.with_columns(
            [
                (
                    pl.col("Vehicle_Title").str.split(
                        by=" ").list.get(0).cast(pl.Int64)
                ).alias("Vehicle_Year"),
                (pl.col("Vehicle_Title").str.split(by=" ").list.get(1)).alias(
                    "Vehicle_Model"
                ),
            ]
        )
        .filter(pl.col("Vehicle_Year").is_in(vehicle_years))
        .select(["Review_Title", "Review", "Rating", "Vehicle_Year", "Vehicle_Model"])
        .sort(["Vehicle_Model", "Rating"])
        .collect()
    )

    # Create ids, documents, and metadatas data in the format chromadb expects
    ids = [f"review{i}" for i in range(car_review_db_data.shape[0])]
    documents = car_review_db_data["Review"].to_list()
    metadatas = car_review_db_data.drop("Review").to_dicts()

    return {"ids": ids, "documents": documents, "metadatas": metadatas}


def build_chroma_collection(
    chroma_path: pathlib.Path,
    collection_name: str,
    embedding_func_name: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    distance_func_name: str = "cosine",
    ):
    """Create a ChromaDB collection"""

    chroma_client = chromadb.PersistentClient(chroma_path)

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_func_name
    )

    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_func,
        metadata={"hnsw:space": distance_func_name},
    )

    document_indices = list(range(len(documents)))

    for batch in batched(document_indices, 166):
        start_idx = batch[0]
        end_idx = batch[-1] + 1  # Fix the end_idx to include the last document

        collection.add(
            ids=ids[start_idx:end_idx],
            documents=documents[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx],
        )

# try:
#     chroma_car_reviews_dict = prepare_car_reviews_data(DATA_PATH)
# except:
#     print("Error preparing car reviews data")


# t0 = time.time()
# print(f"start building chroma at {t0}")
# build_chroma_collection(
#     CHROMA_PATH,
#     COLLECTION_NAME,
#     EMBEDDING_FUNC_NAME,
#     chroma_car_reviews_dict["ids"],
#     chroma_car_reviews_dict["documents"],
#     chroma_car_reviews_dict["metadatas"]
# )
# t1 = time.time()
# print(f"finished building chroma at {t1}")
# total = t1-t0
# print(total)
try:
  client = chromadb.PersistentClient(CHROMA_PATH)
  embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_FUNC_NAME
    )
  collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)
except:
    print("Error connecting to chroma or PATH not found")

# Retrieve all reviews in batches to create a comprehensive context
reviews = []
batch_size = 1000  # Adjust the batch size based on the dataset size and memory limits
current_start = 0

print("Begin to query the collection.")
input1 = "Postive reviews about Volvo"
reviews = collection.query(
    query_texts=[input1],
    n_results=100,
    include=["documents"],
    where={"Rating": {"$gte": 1}},
)

reviews_str = ",".join(reviews["documents"][0])
print("Finished querying the collection.")

question = "Recommend me a Volvo model if I want to prioritize comfort."

context = f"""
You are a customer success employee at a large car dealership. 
Use the following car reviews to answer questions: {reviews_str}. 
You may summarize or rank the data if the question asked you to do so.
You can only use the reviews provided to you to answer the question.
You may start your answers with a short summary consists of 1 to 3 sentences. 
You should provide snippets from top 5 reviews you used to answer as proofs.  
You should not generate new reviews. 
Ensure the reviews you used support your argument. 
"""

prompt = context + "\n" + question

response = model.generate_content(prompt)
print("---------------------------------------------------------\nGemini:\n"+response.text)
