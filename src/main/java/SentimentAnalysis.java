import opennlp.tools.doccat.*;
import opennlp.tools.util.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class SentimentAnalysis {
    private static ArrayList<String> tags; // ArrayList of correct tags (sentiments).
    private static ArrayList<String> resultTags; // ArrayList of predicted tags (sentiments).
    private static ArrayList<String> testArray; // ArrayList of test Tweets.

    private static DoccatModel model; // Sentiment analysis model
    private static DocumentCategorizerME categorizer; // Sentiment catagorizer

    // Possible testing results in [TruthPrediction] format. We define a testing
    // result as a (True Sentiment, Predicted Sentiment) pair.
    private static String[] result =
            {"PositivePositive", "NeutralPositive", "NegativePositive",
                    "PositiveNeutral", "NeutralNeutral", "NegativeNeutral",
                    "PositiveNegative", "NeutralNegative", "NegativeNegative"};
    private static int[] count = new int[9];

    // FEATURE 2: NLP model training on Bitcoin tweets using OpenNLP library

    // Fetches the training data from the text file. Creates a NLP model using
    // OpenNLP library and trains it on the data.
    // Sources:
    // https://opennlp.apache.org/docs/1.9.4/manual/opennlp.html
    // https://stackoverflow.com/questions/42908442/opennlp-categorize-content-return-always-first-category
    public static void trainModel() {
        try {
            System.out.println("Training model...");
            // Set up input stream
            InputStreamFactory inputFactory = new MarkableFileInputStreamFactory(new File("src/main/trainset.txt"));
            ObjectStream<String> lineStream = new PlainTextByLineStream(inputFactory, "UTF-8");
            ObjectStream<DocumentSample> sampleStream = new DocumentSampleStream(lineStream);

            // Create a sentiment model and train it on the trainset.txt file
            model = DocumentCategorizerME.train("en", sampleStream, TrainingParameters.defaultParams(), new DoccatFactory());
            categorizer = new DocumentCategorizerME(model);


        } catch (IOException exception) {
            // Failed to read or parse training data, training failed
            exception.printStackTrace();
        }
    }


    /*___________________________________________________________________________________________*/
    // FEATURE 3: NLP model testing on Bitcoin tweets using OpenNLP library

    // Updates the count of possible testing results
    public static void updateCount(int index) {

        // Concatenate true sentiment with predicted sentiment
        String combinedSentiment = tags.get(index) + resultTags.get(index);

        // Update the count in the frequency array
        for (int i = 0; i < 9; i++) {
            if (result[i].equals(combinedSentiment)) {
                count[i]++;
            }
        }
    }

    // Stores the test predictions in a Confusion Matrix
    public static int[][] createResult() {
        if (tags.size() != resultTags.size()) {
            System.out.println("Number of tags don't match");
        }

        // Reset counts in case we want to use same trained model with
        // different tests
        for (int i = 0; i < 9; i++) {
            count[i] = 0;
        }
        for (int j = 0; j < tags.size(); j++) {
            updateCount(j);
        }

        // Put the result frequencies from the count array into a 3x3
        // matrix
        int[][] confusionMatrix = new int[3][3];
        int index = 0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                confusionMatrix[i][j] = count[index];
                index++;
            }

        return confusionMatrix;
    }

    // Fetches the test data. Tests the trained NLP model on it
    // and stores the results in an ArrayList.
    public static void testModel() throws FileNotFoundException {
        System.out.println("Testing model...");
        createArrays();
        String[] tests = testString();
        resultTags = new ArrayList<String>();

        // Iterate through test strings and assign a category
        for (int i = 0; i < tests.length; i++) {
            double[] outcomes = categorizer.categorize(tests[i].split(" "));
            String category = categorizer.getBestCategory(outcomes);

            // Capitalizes sentiment
            char first = Character.toUpperCase(category.charAt(0));
            String sentiment = first + category.substring(1);

            resultTags.add(sentiment);
        }

    }

    // Parses the testset.txt and testsettag.txt files and puts each line
    // of these files into the corresponding ArrayList.
    public static void createArrays() throws FileNotFoundException {
        File testFile = new File("src/main/testset.txt");
        File tagFile = new File("src/main/testsettag.txt");
        Scanner testScan = new Scanner(testFile);
        Scanner tagScan = new Scanner(tagFile);
        testArray = new ArrayList<String>();
        tags = new ArrayList<String>();

        while (testScan.hasNextLine() && tagScan.hasNextLine()) {

            // Read a line of the test and tag text files
            String testString = testScan.nextLine();
            String tagString = tagScan.nextLine();

            // Preprocess and lemmatize the line
            testString = Preprocessing.preprocessString(testString);
            testString = Preprocessing.lemmatizeString(testString);

            // Capitalize tags
            char first = Character.toUpperCase(tagString.charAt(0));
            tagString = first + tagString.substring(1);

            testArray.add(testString);
            tags.add(tagString);
        }

    }

    // Converts the testArray ArrayList into an array of fixed size.
    // Compatable with OpenNLP catagorizer parameter type.
    public static String[] testString() {
        String[] testList = new String[testArray.size()];
        int i = 0;
        for (String a : testArray) {
            testList[i] = a;
            i++;
        }
        return testList;
    }

    // Prints Confusion Matrix to the terminal
    public static void printMatrix(int[][] mat) {
        int rows = mat.length;
        int columns = mat[0].length;
        String str = "P/T \t";
        str += "Pos" + "\t" + "Neu" + "\t" + "Neg" + "\t";
        System.out.println(str);
        str = "Pos |\t";

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                str += mat[i][j] + "\t";
            }
            System.out.println(str + "|");
            if (i == 0) str = "Neu |\t";
            else if (i == 1) str = "Neg |\t";
        }
    }

    // Allows users to input their own sentences in the terminal
    // to test the model.
    public static void individualSentiment() {
        Scanner scan = new Scanner(System.in);
        String input = "";
        System.out.println("Type your own Bitcoin tweets to check the sentiment!");
        System.out.println("Type 'EXIT' on a new line to stop the program");
        while (!input.equals("EXIT")) {
            input = scan.nextLine();
            if (input.equals("EXIT")) break;

            // Preprocess the line
            input = Preprocessing.preprocessString(input);
            input = Preprocessing.lemmatizeString(input);

            // Assign the line a sentiment
            double[] outcomes = categorizer.categorize(input.split(" "));
            String category = categorizer.getBestCategory(outcomes);

            // Capitalize tag
            char first = Character.toUpperCase(category.charAt(0));
            category = first + category.substring(1);
            System.out.println(category);
        }
    }

    // Preprocesses input data, splits it into train and test sets, outputs
    // a Confusion Matrix and an accuracy score.
    public static void main(String[] args) throws IOException {
        System.out.println("Do not worry about the red logger implementation message...");
        Preprocessing.Preprocess("src/main/bitcointweets.csv", Integer.parseInt(args[0]));
        trainModel();
        testModel();
        int[][] mat = createResult();
        int total = 0;
        int correct = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                total += mat[i][j];
                if (i == j) correct += mat[i][j];
            }
        }
        System.out.println("Rows of Confusion Matrix correspond to predictions, columns correspond to true tags." +
                "\nDiagonal entries are the counts of correct predictions for each tag.");
        printMatrix(mat);
        System.out.println("Accuracy Score: " + ((double) correct / total));
        individualSentiment();
    }
}
