import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import tech.tablesaw.api.Row;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Properties;

// FEATURE 1: Methods for data frame management and String processing

// Interface to drop square brackets and quotes in a String
interface DropBrackets {
    String dropBrackets(String str);
}

// Interface to remove redundant whitespace in a String
interface WhitespaceRemover {
    String removeWhitespace(String str);
}

// Interface to remove emoji in a String
interface EmojiRemover {
    String removeEmoji(String str);
}

// Interface to remove Twitter mentions, ie. words that begin with
// '@' in a String.
interface MentionRemover {
    String removeMentions(String str);
}

// Interface to remove web links in a String
interface LinkRemover {
    String removeLinks(String str);
}

// Interface to remove special characters in a String
interface SpecialRemover {
    String removeSpecialChar(String str);
}

// Interface to lemmatize a String
interface Lemmatizer {
    String lemmatize(String str);
}

public class Preprocessing {

    // Create  global lambda functions for removing whitespace, emoji, Twitter mentions,
    // links and special characters.
    public static WhitespaceRemover lambdaWhitespace = (String str) -> str.trim().replaceAll("\\s{2,}", " ").replaceAll("\\R+", " ");
    public static EmojiRemover lambdaEmoji = (String str) -> str.replaceAll("[^\\p{L}\\p{N}\\p{P}\\p{Z}]", "");
    public static MentionRemover lambdaMentions = (String str) -> str.replaceAll("@.*?\\s+", "");
    public static LinkRemover lambdaLinks = (String str) -> str.replaceAll("http.*\s", " ").replaceAll("http.*", " ");
    public static SpecialRemover lambdaSpecial = (String str) -> str.replaceAll("['.`~|<>,/:;-=+_&^%()]", "").replace("\"", "");

    // Reads data from a CSV file and returns it as a formatted data frame.
    // Throws IOException if the file path does not lead to a .csv file
    // tablesaw API is used throughout for data frame management.
    // Source:
    // https://github.com/jtablesaw/tablesaw
    public static Table readCSV(String filePath) throws IOException {
        File file = new File(filePath);
        return Table.read().csv(file);
    }

    // Cleans a String by removing redundant characters, white space and emoji.
    public static String preprocessString(String text) {

        // Map lambda functions to string and make the string lowercase.
        text = lambdaWhitespace.removeWhitespace(text);
        text = lambdaEmoji.removeEmoji(text);
        text = lambdaMentions.removeMentions(text);
        text = lambdaLinks.removeLinks(text);
        text = lambdaSpecial.removeSpecialChar(text);
        text = text.toLowerCase();

        return text;
    }

    // Cleans a String column of the data frame by removing redundant characters,
    // white space and emoji. Throws IllegalArgumentException if provided
    // column does not exist in the data frame.
    public static void preprocessColumn(Table dataFrame, String columnName) {
        if (!dataFrame.containsColumn(columnName))
            throw new IllegalArgumentException("Column does not exist in data frame.");

        // Map lambda functions to column one by one and turn column strings to lower case
        StringColumn cleanedColumn = dataFrame.column(columnName).asStringColumn();
        cleanedColumn = cleanedColumn.map(lambdaWhitespace::removeWhitespace, StringColumn::create);
        cleanedColumn = cleanedColumn.map(lambdaEmoji::removeEmoji, StringColumn::create);
        cleanedColumn = cleanedColumn.map(lambdaMentions::removeMentions, StringColumn::create);
        cleanedColumn = cleanedColumn.map(lambdaLinks::removeLinks, StringColumn::create);
        cleanedColumn = cleanedColumn.map(lambdaSpecial::removeSpecialChar, StringColumn::create);
        cleanedColumn = cleanedColumn.lowerCase().setName(columnName);

        // Replace old column with the cleaned one
        dataFrame.removeColumns(columnName);
        dataFrame.addColumns(cleanedColumn);
    }

    // Replaces words in a String with their simple dictionary form using
    // Stanford CoreNLP's lemmatization API. Returns lemmatized String.
    // Source:
    // https://stanfordnlp.github.io/CoreNLP/lemma.html
    public static String lemmatizeString(String str) {
        Properties properties = new Properties();
        properties.setProperty("annotators", "tokenize, ssplit, pos, lemma");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(properties);
        CoreDocument document = new CoreDocument(str);
        pipeline.annotate(document);
        StringBuilder builder = new StringBuilder();
        for (CoreLabel token : document.tokens()) {
            String word = token.lemma() + " ";
            builder.append(word);
        }
        return builder.toString();
    }

    // Lemmatizes a String column of a data frame. Throws IllegalArgumentException
    // if provided column does not exist in the data frame.
    // Source:
    // https://stanfordnlp.github.io/CoreNLP/lemma.html
    public static void lemmatizeColumn(Table dataFrame, String columnName) {
        if (!dataFrame.containsColumn(columnName))
            throw new IllegalArgumentException("Column does not exist in data frame.");

        // Create lambda function for lemmatization
        Lemmatizer lambdaLemma = (String str) -> lemmatizeString(str);

        // Map lambda function to column
        StringColumn column =
                dataFrame.column(columnName).asStringColumn().map(lambdaLemma::lemmatize, StringColumn::create);
        dataFrame.replaceColumn(columnName, column);
    }

    // Turns Tag column into a clean form. Throws IllegalArgumentException if
    // provided column does not exist in the data frame.
    public static void prepareTag(Table dataFrame, String columnName) {
        if (!dataFrame.containsColumn(columnName))
            throw new IllegalArgumentException("Column does not exist in data frame.");

        // Create lambda function for getting rid of brackets around tag
        DropBrackets lambdaBrackets = (String str) -> str.substring(2, str.length() - 2);

        // Map lambda function to column
        StringColumn column =
                dataFrame.column(columnName).asStringColumn().map(lambdaBrackets::dropBrackets, StringColumn::create);

        // Capitalize tag column
        column = column.capitalize().setName(columnName);
        dataFrame.replaceColumn(columnName, column);
    }

    // Randomly splits data frame into a training set and a test set for the
    // natural language processing model. p is the proportion of data frame
    // rows that will be in the training set. Standardizes the train and test
    // sets to be used with Apache OpenNLP API. Returns array with train and
    // test sets. Throws IllegalArgumentException if proportion is invalid.
    public static Table[] trainTestSplit(Table dataFrame, String columnName, String tagName, double p) {
        if (p >= 1)
            throw new IllegalArgumentException("Fraction must be between 0 and 1");

        // Split data frame
        Table[] trainTest = dataFrame.sampleSplit(p);
        Table train = trainTest[0];
        Table test = trainTest[1];

        for (Row row : train) {
            String trainString = row.getString(tagName) + " " + row.getString(columnName);
            row.setString(columnName, trainString);
        }
        train.retainColumns(columnName);

        return new Table[]{train, test};
    }

    // Outputs the training data frame to a text file.
    public static void trainToTXT(Table dataFrame, String columnName, String fileName) {
        StringBuilder sb = new StringBuilder();

        // Adds non-empty row entries to StringBuilder, one by one
        sb.append(dataFrame.column(columnName).asStringColumn().get(0));
        for (int i = 1; i < dataFrame.column(columnName).size(); i++) {
            String str = dataFrame.column(columnName).asStringColumn().get(i);
            if (!str.equals("neutral ") && !str.equals("positive ") && !str.equals("negative "))
                sb.append("\n" + dataFrame.column(columnName).asStringColumn().get(i));
        }

        // Print StringBuilder to text file
        try (PrintWriter out = new PrintWriter("src/main/" + fileName)) {
            out.println(sb);
        } catch (FileNotFoundException exception) {
            exception.printStackTrace();
        }
    }

    // Outputs the test strings and their corresponding tags to separate
    // text files, in order to simulate real testing conditions. Text
    // file with one message per line is a reasonable "real world" input.
    public static void testToTXT(Table dataFrame, String columnTest, String columnTag, String fileTest, String fileTag) {
        StringBuilder sbTest = new StringBuilder();
        StringBuilder sbTag = new StringBuilder();

        // Remove whitespace and line breaks so that the strings will be one
        // per line
        StringColumn newColumn =
                dataFrame.column(columnTest).asStringColumn().map(lambdaWhitespace::removeWhitespace, StringColumn::create);
        dataFrame.removeColumns(columnTest);
        dataFrame.addColumns(newColumn);

        // Adds non-empty row entries to StringBuilders, one by one
        sbTest.append(dataFrame.column(columnTest).asStringColumn().get(0));
        sbTag.append(dataFrame.column(columnTag).asStringColumn().get(0));
        for (int i = 1; i < dataFrame.column(columnTest).size(); i++) {
            if (!dataFrame.column(columnTest).asStringColumn().get(i).equals("  ")) {
                sbTest.append("\n" + dataFrame.column(columnTest).asStringColumn().get(i));
                sbTag.append("\n" + dataFrame.column(columnTag).asStringColumn().get(i));
            }
        }

        // Prints test data and tags in separate text files
        try (PrintWriter out = new PrintWriter("src/main/" + fileTest)) {
            out.println(sbTest);
        } catch (FileNotFoundException exception) {
            exception.printStackTrace();
        }
        try (PrintWriter out = new PrintWriter("src/main/" + fileTag)) {
            out.println(sbTag);
        } catch (FileNotFoundException exception) {
            exception.printStackTrace();
        }
    }

    // Preprocesses the first numTweets bitcoin tweets csv file to be ready
    // for NLP model training and testing.
    public static void Preprocess(String filePath, int numTweets) throws IOException {
        System.out.println("Reading input data...");
        Table dataFrame = readCSV(filePath);
        dataFrame = dataFrame.dropRowsWithMissingValues();

        // Only keep relevant columns
        dataFrame.retainColumns("Tweet", "Tag");
        if (numTweets > dataFrame.column("Tweet").size())
            throw new IllegalArgumentException("Input is larger than length of data frame");
        dataFrame = dataFrame.first(numTweets);
        prepareTag(dataFrame, "Tag");

        // Create train and test data frames
        System.out.println("Splitting data into train and test sets...");
        Table[] trainTest = trainTestSplit(dataFrame, "Tweet", "Tag", 0.8);
        Table train = trainTest[0];
        Table test = trainTest[1];

        // Preprocess only the train data to make learning easier for
        // the model! We want the test data to be similar to a real
        // world input
        System.out.println("Preprocessing train data...");
        preprocessColumn(train, "Tweet");
        System.out.println("Lemmatizing train data...");
        System.out.println("(This may take a while)");
        lemmatizeColumn(train, "Tweet");
        train.dropRowsWithMissingValues();
        test.dropRowsWithMissingValues();

        // Output the preprocessed data to text files
        System.out.println("Creating input text files...");
        trainToTXT(train, "Tweet", "trainset.txt");
        testToTXT(test, "Tweet", "Tag", "testset.txt", "testsettag.txt");
    }

    // Testing preprocessColumn and lemmatizeColumn
    public static void main(String[] args) throws IOException {

        // Create two columns, A has redundant verbosity that will be cleaned up
        // and B is a corner case with empty values only. Add columns to a data frame
        String[] stringsA = {"Hi @@_ are YOU seeking HELP?", "HEY!!111 \n <333", "i AM feeling \n hApPY"};
        String[] stringsB = {"", "", ""};
        StringColumn columnA = StringColumn.create("A", stringsA);
        StringColumn columnB = StringColumn.create("B", stringsB);
        Table dataFrame = Table.create(columnA, columnB);

        // Print initial state
        System.out.println("Initial columns:");
        System.out.println(columnA.print());
        System.out.print(columnB.print());

        // Preprocess the columns and print new state
        preprocessColumn(dataFrame, "A");
        preprocessColumn(dataFrame, "B");
        System.out.println("After preprocessing:");
        System.out.println(dataFrame.column("A").print());
        System.out.print(dataFrame.column("B").print());

        // Lemmatize the columns and print new state
        lemmatizeColumn(dataFrame, "A");
        lemmatizeColumn(dataFrame, "B");
        System.out.println("After lemmatization:");
        System.out.println(dataFrame.column("A").print());
        System.out.print(dataFrame.column("B").print());
    }
}

