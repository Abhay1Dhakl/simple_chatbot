# Simple Agent Chatbot

A Flask-based chatbot application that uses ChromaDB for vector storage and retrieval, with integration to Azure OpenAI/Meta-Llama models for intelligent responses.

##  Features

- **Vector Database**: ChromaDB for efficient document storage and retrieval
- **Semantic Search**: Sentence Transformers for query vectorization
- **LLM Integration**: Azure OpenAI/Meta-Llama model integration
- **REST API**: Flask-based API for chatbot interactions
- **Docker Support**: Containerized deployment ready

##  Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning)

##  Installation & Setup

### 1. Clone the Repository (Optional)
```bash
git clone https://github.com/Abhay1Dhakl/simple_chatbot.git
cd simple_chatbot
```

### 2. Create Virtual Environment

#### On Windows:
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirement.txt
```

### 4. Environment Configuration

Create a `.env` file in the root directory and add your configuration:

```env
OPENAI_BASE_URL=your_openai_base_url_here
OPENAI_API_KEY=your_openai_api_key_here
```

**Note**: Replace the placeholder values with your actual API credentials.

### 5. Database Setup

The application uses ChromaDB for vector storage. The database will be automatically created in the `chroma_db/` directory when you first run the application.

If you have existing data to embed, you can use the provided utility scripts:
- `embedding_data.py` - For embedding general data
- `embedding_csv_file.py` - For embedding CSV data
- `data_conversion.py` - For data format conversion

##  Running the Application

### Development Mode

1. **Activate Virtual Environment** (if not already activated):
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Run the Flask Application**:
   ```bash
   python app.py
   ```

3. **Access the API**:
   - The application will start on `http://localhost:5000`
   - Use the `/query` endpoint to interact with the chatbot

### Production Mode (Docker)

1. **Build Docker Image**:
   ```bash
   docker build -t simple-agent-chatbot .
   ```

2. **Run Docker Container**:
   ```bash
   docker run -p 5000:5000 simple-agent-chatbot
   ```

## 📡 API Usage

### Query Endpoint

**POST** `/query`

**Request Body**:
```json
{
    "query": "Your question here"
}
```

**Response**:
```json
{
    "response": "AI generated response",
    "status": "success"
}
```

### Example using curl:
```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about laptops"}'
```

### Example using Python:
```python
import requests

response = requests.post('http://localhost:5000/query', 
                        json={'query': 'Tell me about laptops'})
print(response.json())
```

##  Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_BASE_URL` | Base URL for OpenAI API | Yes |
| `OPENAI_API_KEY` | API key for OpenAI | Yes |

### Model Configuration

The application is configured to use:
- **Embedding Model**: `all-MiniLM-L6-v2` (SentenceTransformers)
- **LLM Model**: Configurable via environment variables

##  Development

### Adding New Data

1. Place your data files in the project directory
2. Use the appropriate embedding script:
   - For CSV files: `python embedding_csv_file.py`
   - For general data: `python embedding_data.py`

### Customizing the Model

You can modify the model configuration in `app.py`:
- Change the embedding model in line with `SentenceTransformer('model-name')`
- Update the LLM configuration in the environment variables

##  Docker Deployment

The application includes a `dockerfile` for easy deployment:

```dockerfile
FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install -r requirement.txt
CMD ["python", "app.py"]
```

##  Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirement.txt
   ```

2. **ChromaDB Connection Issues**: Check if the `chroma_db/` directory exists and is accessible

3. **API Key Issues**: Verify your `.env` file contains correct API credentials

4. **Port Already in Use**: Change the port in `app.py` or kill the process using the port

### Debug Mode

Run the application in debug mode for detailed error messages:
```bash
export FLASK_DEBUG=1  # Linux/Mac
set FLASK_DEBUG=1     # Windows
python app.py
```

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation

##  Updates

To update the project:
1. Pull latest changes: `git pull origin main`
2. Update dependencies: `pip install -r requirement.txt --upgrade`
3. Restart the application

---

**Happy Coding! 🎉**
