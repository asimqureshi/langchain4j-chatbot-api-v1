# Chatbot API powered by LangChain4j

A Spring Boot-based chatbot application powered by LangChain4j, OpenAI, and PostgreSQL with pgvector for semantic search and embeddings.

## 🚀 Features

- **AI-Powered Chat**: Interactive chat interface with OpenAI GPT-4
- **Document Embedding**: Upload and embed large text documents for context-aware responses
- **Semantic Search**: PostgreSQL with pgvector for efficient similarity search
- **Modern UI**: Beautiful, responsive web interface with real-time chat
- **File Upload**: Support for text files (.txt, .md, .csv) for embedding
- **Embedding Management**: Clear all embeddings functionality

## 🏗️ Architecture

- **Backend**: Spring Boot 3.4.2 with Java 21
- **AI Models**: OpenAI GPT-4 and OpenAI Text Embedding 3 Small
- **Database**: PostgreSQL with pgvector extension
- **Frontend**: Vanilla JavaScript with modern CSS
- **Embedding Store**: LangChain4j PgVector Embedding Store

## 📋 Prerequisites

- Java 21 or higher
- Maven 3.6+
- PostgreSQL 12+ with pgvector extension
- OpenAI API key

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd chatbot-api-v1
```

### 2. Database Setup

#### Install pgvector Extension

```sql
-- Connect to your PostgreSQL database
CREATE EXTENSION IF NOT EXISTS vector;
```

#### Create Table

```sql
create table documents (
  id bigserial not null,
  text text null,
  metadata jsonb null,
  embedding vector null,
  embedding_id uuid not null,
  constraint documents_pkey primary key (id, embedding_id),
  constraint documents_embedding_id_key unique (embedding_id)
);
```

### 3. Environment Variables

Create a `.env` file in the project root or set the following environment variables:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# PostgreSQL Configuration (optional - can be configured in application.yml)
PGVECTOR_HOST=localhost
PGVECTOR_PORT=5432
PGVECTOR_DATABASE=postgres
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=your_password_here
```
## 🚀 Running the Application

```bash
# Build the project
mvn clean install

# Run the application
mvn spring-boot:run
```

The application will start on `http://localhost:8080`

## 📱 Usage

### Web Interface

1. **Chat Interface** (`http://localhost:8080/index.html`)
   - Interactive chat with the AI assistant
   - Real-time responses with typing indicators
   - Modern, responsive design

2. **Embedding Interface** (`http://localhost:8080/embed.html`)
   - Upload text files or paste large text blocks
   - Process embeddings for semantic search
   - Clear all embeddings functionality

### API Endpoints

#### Chat Endpoint
```http
POST /chatbot/chat
Content-Type: application/json

{
  "message": "Your question here"
}
```

#### Embed Endpoint
```http
POST /chatbot/embed
Content-Type: application/json

{
  "text": "Large text block to embed"
}
```

#### Clear Embeddings Endpoint
```http
DELETE /chatbot/embed
```

## 🏗️ Project Structure

```
langchain-chatbot-api-v1/
├── src/
│   ├── main/
│   │   ├── java/lab/maq/langchain/chatbot/
│   │   │   ├── ChatBotConfiguration.java    # Spring configuration
│   │   │   ├── Main.java                    # Application entry point
│   │   │   └── impl/
│   │   │       ├── ChatBotController.java   # REST endpoints
│   │   │       ├── ChatBotService.java      # Business logic
│   │   │       ├── ChatModel.java           # Chat request/response model
│   │   │       └── EmbedModel.java          # Embedding request model
│   │   └── resources/
│   │       ├── application.yml              # Application configuration
│   │       └── static/
│   │           ├── index.html               # Chat interface
│   │           └── embed.html               # Embedding interface
│   └── test/
└── pom.xml                                  # Maven dependencies
```

## 🔍 Key Components

### ChatBotService
- Handles document embedding and chunking
- Manages semantic search queries
- Processes chat interactions with context

### Embedding Store
- Uses PostgreSQL with pgvector for vector storage
- Supports similarity search for context retrieval
- Handles large text document processing

### Frontend
- Modern, responsive design
- Real-time chat interface
- File upload and embedding management
- Navigation between chat and embed pages

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Ensure PostgreSQL is running
   - Verify pgvector extension is installed
   - Check database credentials in `application.yml`

2. **OpenAI API Errors**
   - Verify your API key is correct
   - Check API key permissions
   - Ensure sufficient API credits

3. **Port Already in Use**
   - Change the port in `application.yml`
   - Or kill the process using the port

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain4j](https://github.com/langchain4j/langchain4j) for the Java AI framework
- [OpenAI](https://openai.com/) for the AI models
- [pgvector](https://github.com/pgvector/pgvector) for PostgreSQL vector operations
- [Spring Boot](https://spring.io/projects/spring-boot) for the application framework 
