import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MyDataReader {
    List<String> documents = new ArrayList<String>();

    public List<String> read(String filename) throws IOException {

        BufferedReader readin = new BufferedReader(new FileReader(filename));
        if (readin == null) {
            System.out.println("Input file has problem...");
            System.exit(0);
        }

        String lineString = new String();
        while ((lineString = readin.readLine()) != null) {
            documents.add(lineString);
        }
        readin.close();

        return documents;
    }
}
