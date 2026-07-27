// Example: Java client using HttpClient (Java 11+)
// mvn dependency not required — standard library only.

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;

public class ForecastClient {
    public static void main(String[] args) throws Exception {
        String body = Files.readString(Path.of("examples/sample_request.json"));
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create("http://localhost:8000/forecast"))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(body))
            .build();
        HttpClient client = HttpClient.newHttpClient();
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        System.out.println(response.statusCode());
        System.out.println(response.body());
    }
}
