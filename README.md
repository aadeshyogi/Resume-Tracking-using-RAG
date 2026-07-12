# Resume Tracking using RAG

A simple app that lets you search resumes using natural language instead of keywords. Built using Retrieval-Augmented Generation (RAG), so it understands multi-condition queries like "candidates who know Python and graduated in 2024" and only returns full matches.

## Tech Stack

- **Streamlit** – UI
- **LangChain** – loading and splitting PDF resumes
- **ChromaDB** – vector database
- **OpenAI (gpt-4o-mini + embeddings)** – answering queries

## How to Run

1. Create a virtual environment and install dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. Add your OpenAI API key in a `.env` file
   ```
   OPENAI_API_KEY=your_key_here
   ```

3. Put your resume PDFs inside a `Resumes` folder
   (update the `resume_folder` path in `app.py` if needed)

4. Run the app
   ```bash
   streamlit run app.py
   ```

## How It Works

1. Loads all PDF resumes
2. Splits them into chunks
3. Converts chunks into embeddings and stores them in ChromaDB
4. On a query, retrieves the most relevant chunks using MMR search
5. Sends the retrieved context to GPT-4o-mini, which gives a grounded answer using only that context

## Note

Don't upload your `.env` file or `Resumes` folder to GitHub — add them to `.gitignore`.

## Credits

Built during a learning project at Innomatics Research Labs, under the guidance of Nagaraju Ekkirala and Mohammad Afroz.
