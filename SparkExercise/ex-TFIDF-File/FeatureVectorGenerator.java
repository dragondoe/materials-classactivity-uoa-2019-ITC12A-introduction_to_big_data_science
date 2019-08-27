import net.reduls.igo.Morpheme;
import net.reduls.igo.Tagger;

import java.io.IOException;
import java.util.*;

public class FeatureVectorGenerator {
    Tagger tagger = null;

    public Map<String, double[]> generateTFIDFVectors(List<String> documents) throws IOException {

        // List<word>
        List<String> words = new ArrayList<String>();

        // Map<document, Map<word, tf>>
        Map<String, Map<String, Integer>> tfMap = new HashMap<String, Map<String, Integer>>();

        // Map<word, df>
        Map<String, Integer> dfMap = new HashMap<String, Integer>();

        // For Igo Mophological Analyzer
        // tagger = new Tagger("/Users/Hikaru/Desktop/SnippetDataGenerator/src/main/java/org/hikaru/generator/ipadic");
        tagger = new Tagger("./ipadic");

        for(String document : documents){

            List<String> parseResult = parse(document);

            Map<String, Integer> docTFMap = new HashMap<String, Integer>();
            for(String word : parseResult){

                // TF
                if(docTFMap.containsKey(word)){
                    int tf = docTFMap.get(word);
                    docTFMap.put(word, tf+1);
                } else{
                    docTFMap.put(word, 1);
                }

                // Word list
                if(!words.contains(word)){
                    words.add(word);
                }
            }

            tfMap.put(document, docTFMap);

            // DF
            for(String word : docTFMap.keySet()){
                if(dfMap.containsKey(word)){
                    int df = dfMap.get(word);
                    dfMap.put(word, df+1);
                } else{
                    dfMap.put(word, 1);
                }
            }
        }


        Map<String, double[]> featureVectors = new HashMap<String, double[]>();
        for(String document : documents) {

            // Map<word, tfidf>
            Map<String, Double> docTFIDFMap = new HashMap<String, Double>();

            // Words TF-IDF
            Map<String, Integer> docTFMap = tfMap.get(document);
            for (String word : docTFMap.keySet()) {

                double tf = (double) docTFMap.get(word);
                double df = (double) dfMap.get(word);
                double N = (double) documents.size();

                double tfidf = tf * Math.log(N / df);

                docTFIDFMap.put(word, tfidf);
            }

            // FV
            double[] featureVector = new double[words.size()];
            for (int i = 0; i < words.size(); i++) {
                String word = words.get(i);
                if (docTFIDFMap.containsKey(word)) {
                    featureVector[i] = docTFIDFMap.get(word);
                } else {
                    featureVector[i] = 0;
                }
            }
            featureVectors.put(document, featureVector);
        }

        // For test
        // writeWord(words);

        return featureVectors;
    }

    private List<String> parse(String str){
        List<String> result = new ArrayList<String>();

        for(Morpheme m : tagger.parse(str))
            result.add(m.surface);

        return result;
    } // end of parse

    private void writeWord(List<String> words){
        // List of Words
        System.out.println("=== List of Words ===");
        System.out.print("(");
        for(int i=0; i < words.size(); i++){
            System.out.print(words.get(i));
            if(i != words.size()-1){
                System.out.print(", ");
            }
        }
        System.out.println(")");
        System.out.println("");
    }


}
