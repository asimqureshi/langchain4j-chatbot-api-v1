package lab.maq.langchain.chatbot.impl;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
@Service
public class ChatBotService {

    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> embeddingStore;
    private final ChatModel chatModel;
    private final Map<String, String> conversationHistory = new HashMap<>();

    /**
     * Processes and embeds text into the vector store for semantic search.
     * Splits large text into chunks and converts each chunk to vector embeddings.
     * 
     * @param text The text content to be embedded
     */
    public void embed(String text) {

        // Split the input text into smaller chunks for better processing
        // Parameters: 500 characters max per chunk, 50 characters overlap between chunks
        List<TextSegment> chunks = DocumentSplitters.recursive(500, 50)
                .split(Document.from(text));

        // Process each text chunk: convert to embedding and store in database
        chunks.forEach(chunk -> {
            // Convert the text chunk to a vector embedding using OpenAI
            Response<Embedding> embeddingResponse = embeddingModel.embed(chunk);
            // Store the embedding vector along with the original text in the database
            embeddingStore.add(embeddingResponse.content(), chunk);
        });

    }

    /**
     * Handles chat interactions with context-aware responses.
     * Converts user message to standalone format, searches for relevant context,
     * and generates a response using the retrieved context.
     * 
     * @param userMessage The user's input message
     * @return AI-generated response based on embedded context
     */
    public String chat(String userMessage) {

        // Create a standalone question, stripping out the un-necessary bits
        String standAloneQuestion = createStandAloneMessage(userMessage);
        // Search for relevant context using the processed question
        Optional<String> context = similaritySearch(standAloneQuestion);

        // If no relevant context is found, return a helpful message
        if(context.isEmpty()) {
            return """
                    I don't have any context to form a reply. Please embed some information using the embed link above.
                    """;
        }

        // Create the prompt using the original user message, the retrieved context and the conversation history
        String prompt = createPrompt(userMessage, context.get());
        // Generate the final response using the chat model
        String answer = chatModel.chat(prompt);
        // Save the q/a in history
        conversationHistory.put(userMessage, answer);

        return answer;
    }

    private String createPrompt(String userMessage, String context) {
        // Template for generating the final answer using context and user question
        String promptTemplate = """
                      You are a helpful and enthusiastic support bot who can answer a given question based on  
                      the context provided. Try to find the answer in the context. Also use the conversation history to answer the question. 
                      If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email help@support.com. 
                      Don't try to make up an answer. Always speak as if you were chatting to a friend.
                      conversation_history: {{conversationHistory}}
                      context: {{context}}
                      question: {{question}}
                      answer:
                """;
        // Apply the answer template with user question and retrieved context
        return PromptTemplate.from(promptTemplate)
                .apply(Map.of("question", userMessage,
                        "context", context,
                        "conversationHistory", getConversationHistory()))
                .text();
    }

    private String createStandAloneMessage(String userQuestion) {
        // Template for converting user question to a standalone question format
        String standaloneMessageTemplate = """
                Given a question, convert it to a standalone question. question: {{question}} standalone question:
                """;

        // Apply the template to the user's question to create a standalone version
        String standAloneMessage = PromptTemplate.from(standaloneMessageTemplate)
                .apply(Map.of("question", userQuestion))
                .text();
        // Send the standalone question to the chat model for processing
        return chatModel.chat(standAloneMessage);
    }

    private String getConversationHistory() {
        return conversationHistory.entrySet().stream()
                .map(entry -> "HUMAN: " + entry.getKey() + "\n AI: " + entry.getValue())
                .collect(Collectors.joining("\n"));
    }

    /**
     * Searches for relevant context using semantic similarity.
     * Converts the question to a vector embedding and finds the most similar
     * text segment from the embedding store.
     * 
     * @param question The question to search for context
     * @return Optional containing the most relevant text segment, or empty if no matches found
     */
    private Optional<String> similaritySearch(String question) {

        // Convert the question to a vector embedding for similarity search
        Embedding queryEmbedding = embeddingModel.embed(question).content();
        // Search the embedding store for similar text segments
        EmbeddingSearchResult<TextSegment> embeddingSearchResult = embeddingStore.search(EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .build());

        // If no matches found, return empty optional
        if(embeddingSearchResult.matches().isEmpty()) {
            return Optional.empty();
        }

        // Get the best matching text segment (highest similarity score)
        EmbeddingMatch<TextSegment> embeddingMatch = embeddingSearchResult.matches().get(0);
        // Return the text content of the best match wrapped in Optional
        return Optional.of(embeddingMatch.embedded().text());
    }

    /**
     * Clears all embeddings from the vector store.
     * Removes all stored text segments and their corresponding vector embeddings.
     */
    public void clearEmbeddings() {
        // Remove all stored embeddings from the database
        embeddingStore.removeAll();
    }
}

