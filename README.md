# Ethiopian Research Paper Assistant

A specialized application for retrieving, summarizing, and analyzing Ethiopian research papers through dedicated AI agents and a modern user interface.

## Features

- **Qdrant Vector Database Integration**: Access to a comprehensive collection of Ethiopian research papers
- **Three Specialized Agents**:
  - **Summarization Agent**: Generate concise summaries of research papers
  - **Literature Review Agent**: Create comprehensive literature reviews on specific topics
  - **Deep Research Agent**: Provide in-depth analysis on research questions

## Setup Instructions

1. **Clone this repository**

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Environment Setup**:
   - Create a `.env` file in the root directory with the following variables:
     ```
     QDRANT_API_KEY=your_qdrant_api_key
     QDRANT_URL=https://12bbc0a4-833b-4e01-880e-fa75e8e962f6.europe-west3-0.gcp.cloud.qdrant.io:6333
     QDRANT_COLLECTION=AASTU
     OPENAI_API_KEY=your_openai_api_key
     ```

4. **Run the application**:
   ```
   uvicorn app.main:app --reload
   ```

5. **Access the application**:
   Open your browser and navigate to `http://localhost:8000`

## Project Structure

```
research-assistant/
├── app/
│   ├── api/             # API routes
│   ├── agents/          # Agent implementations
│   ├── models/          # Data models
│   ├── static/          # Static files (CSS, JS)
│   ├── templates/       # HTML templates
│   └── main.py          # Application entry point
├── .env                 # Environment variables
├── README.md            # Project documentation
└── requirements.txt     # Dependencies
```

## Usage

1. **Enter a research query** in the main search bar
2. **Select an agent type** (Summarization, Literature Review, or Deep Research)
3. **View the results** displayed in a clear, organized format
4. **Explore additional features** such as filtering results and viewing metadata

## Technologies Used

- **Backend**: FastAPI, LangChain, Qdrant
- **Frontend**: HTML, CSS, JavaScript
- **Database**: Qdrant Vector Database
