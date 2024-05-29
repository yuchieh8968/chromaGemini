import chromadb
import pathlib
import polars as pl
from more_itertools import batched
from chromadb.utils import embedding_functions
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import time

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

# specify paths and variables
DATA_PATH = "./Airline_review.csv"
CHROMA_PATH = "./flight_review_embeddings"
EMBEDDING_FUNC_NAME = "multi-qa-MiniLM-L6-cos-v1"
COLLECTION_NAME = "flight_reviews"

def extract_year(date_str: str) -> int:
    """Extracts the year from different date formats."""
    months = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
        'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
        'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    try:
        # Handling format "MMM-YY"
        if '-' in date_str:
            month, year_suffix = date_str.split('-')
            year = 2000 + int(year_suffix) if int(year_suffix) < 70 else 1900 + int(year_suffix)
            return year
        # Handling format "MMMYYYY"
        else:
            month = date_str[:3]
            year = int(date_str[3:])
            return year
    except Exception as e:
        print(f"Error parsing date: {date_str}, Error: {e}")
        return None

def prepare_airline_reviews_data(data_path: pathlib.Path, flight_years: list[int] = [2017]):
    """Prepare the airline reviews dataset for ChromaDB"""

    # Define the schema to ensure proper data types are enforced
    dtypes = {
        "Id": pl.Int64,
        "Airline Name": pl.Utf8,
        "Overall_Rating": pl.Float64,
        "Review_Title": pl.Utf8,
        "Review Date": pl.Utf8,
        "Verified": pl.Utf8,
        "Review": pl.Utf8,
        "Aircraft": pl.Utf8,
        "Type Of Traveller": pl.Utf8,
        "Seat Type": pl.Utf8,
        "Route": pl.Utf8,
        "Date Flown": pl.Utf8,
        "Seat Comfort": pl.Float64,
        "Cabin Staff Service": pl.Float64,
        "Food & Beverages": pl.Float64,
        "Ground Service": pl.Float64,
        "Inflight Entertainment": pl.Float64,
        "Wifi & Connectivity": pl.Float64,
        "Value For Money": pl.Float64,
        "Recommended": pl.Utf8
    }

    # Scan the airline reviews dataset(s)
    airline_reviews = pl.scan_csv(
        data_path, 
        dtypes=dtypes, 
        null_values="n",
        infer_schema_length=10000
    )


    # Filter on selected years and handle potential missing data
    airline_review_db_data = (
        airline_reviews.with_columns()
        .filter(pl.col("Verified")=="TRUE")
        .select(["Review_Title", "Review", "Overall_Rating", "Verified", "Airline Name"])
        .sort(["Airline Name", "Overall_Rating"])
        .collect()
    )

    # Create ids, documents, and metadatas data in the format ChromaDB expects
    ids = [f"review{i}" for i in range(airline_review_db_data.shape[0])]
    documents = airline_review_db_data["Review"].to_list()
    metadatas = airline_review_db_data.drop("Review").to_dicts()

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

chroma_car_reviews_dict = prepare_airline_reviews_data(DATA_PATH)

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

client = chromadb.PersistentClient(CHROMA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_FUNC_NAME
    )
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

# Retrieve all reviews in batches to create a comprehensive context
reviews = []
batch_size = 1000  # Adjust the batch size based on the dataset size and memory limits
current_start = 0

print("Begin to query the collection.")
reviews = collection.query(
    query_texts=["Find me some bad reviews about Air Canada"],
    n_results=50,
    include=["documents", "metadatas"],
)

reviews_str = ",".join(reviews["documents"][0])
metadata = str(reviews["metadatas"][0])
print("Finished querying the collection.")

question = "Show me 3 of the most angry and negative reviews."

context = f"""
You are a customer service agent specializes in airline customer reviews. Use the following flight reviews to answer questions: {reviews_str} and its metadatas {metadata}. 
You may summarize or rank the data if the question asked you to do so.
You can only use the reviews and metadats provided to you to answer the question.
You may start your answers with a short summary consists of 3 to 5 sentences. 
You should provide proof from top 5 reviews you used to answer, and list the reviews' metadata.  
You should not generate new reviews and metadata. 
Ensure the reviews you used support your argument. 
"""

prompt = context + "\n" + question

response = model.generate_content(prompt)
print("---------------------------------------------------------\nGemini:\n"+response.text)
