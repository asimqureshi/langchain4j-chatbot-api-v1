package lab.maq.langchain.chatbot;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.http.client.spring.restclient.SpringRestClient;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.pgvector.PgVectorEmbeddingStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import static dev.langchain4j.internal.Utils.getOrDefault;
import static dev.langchain4j.model.openai.OpenAiChatModelName.GPT_4_1_NANO;
import static dev.langchain4j.model.openai.OpenAiEmbeddingModelName.TEXT_EMBEDDING_3_SMALL;

@Configuration
public class ChatBotConfiguration {

    @Value("${openai.api.key}")
    private String openAiApiKey;
    @Value("${pgvector.host}")
    private String host;
    @Value("${pgvector.port}")
    private int port;
    @Value("${pgvector.database}")
    private String database;
    @Value("${pgvector.user}")
    private String username;
    @Value("${pgvector.password}")
    private String password;

    @Bean
    EmbeddingModel embeddingModel() {
        return OpenAiEmbeddingModel.builder()
                .httpClientBuilder(SpringRestClient.builder())
                .modelName(TEXT_EMBEDDING_3_SMALL)
                .apiKey(openAiApiKey)
                .build();
    }

    @Bean
    EmbeddingStore<TextSegment> embeddingStore(EmbeddingModel embeddingModel) {
        return PgVectorEmbeddingStore.builder()
                .host(host)
                .port(port)
                .database(database)
                .user(username)
                .password(password)
                .table("documents")
                .dimension(embeddingModel.dimension())
                .build();
    }

    @Bean
    ChatModel chatModel() {
        return OpenAiChatModel.builder()
                .modelName(GPT_4_1_NANO)
                .httpClientBuilder(SpringRestClient.builder())
                .apiKey(openAiApiKey)
                .build();
    }

}
