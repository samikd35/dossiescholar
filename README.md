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

3. **Set up environment variables**:
   Create a `.env` file with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   QDRANT_COLLECTION=your_collection_name
   ENVIRONMENT=development
   ```

4. **Run the application**:
   ```
   python -m uvicorn app.main:app --reload
   ```

5. **Run tests**:
   ```
   pytest tests/
   ```

## Project Structure

```
├── app
│   ├── agents                # LangGraph-based AI agents
│   ├── models                # Data models and vector store
│   ├── static                # Static files (CSS, JS)
│   └── templates             # Jinja2 templates
├── tests                     # Test files
├── .env                      # Environment variables (not tracked by git)
├── README.md                 # This file
└── requirements.txt          # Project dependencies
```

## Latest Updates

- **LangGraph Migration**: Converted agents from LangChain to LangGraph
- **Error Handling**: Added comprehensive error handling
- **Dependency Management**: Optimized dependency versions
- **Logging**: Added structured logging with Loguru
- **Performance Optimization**: Added caching for agent initialization
- **Testing**: Added unit tests
- **Documentation**: Enhanced code documentation and README

## Environment Configuration

The application supports different environment configurations through the `ENVIRONMENT` environment variable:
- `development`: Enables auto-reload and verbose logging
- `production`: Disables auto-reload and uses optimized settings

## Troubleshooting

If you encounter dependency issues with NumPy or other packages:
```
pip uninstall -y numpy
pip install numpy==1.24.0
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
