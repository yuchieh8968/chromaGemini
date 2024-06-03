# chromaGemini
Utilizes Google's Gemini 1.5 Flash model and ChromaDB to analyze any given dataset and receive a human readable response answering user's questions.
Data Flow:
1. Data is processed and embedded with default embedding function provided by ChromaDB
2. Data is then stored in a collection for later access.
3. Program receives a query against the query, and return top 100 matches.
4. The matches are fed to Gemini with proper pre-prompts to return the most relevant result and summary.