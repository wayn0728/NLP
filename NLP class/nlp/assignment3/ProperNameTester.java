package nlp.assignment3;

import java.util.List;
import java.util.Collection;
import java.util.ArrayList;
import java.util.Map;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileNotFoundException;
import java.io.IOException;

import nlp.assignment2.MostFrequentLabelClassifier;
import nlp.assignment2.MostFrequentLabelClassifier.Factory;
import nlp.classify.*;
import nlp.langmodel.LanguageModel;
import nlp.util.CommandLineUtils;
import nlp.util.Counter;
import nlp.util.Pair;

/**
 * This is the main harness for assignment 2. To run this harness, use
 * <p/>
 * java edu.berkeley.nlp.assignments.ProperNameTester -path ASSIGNMENT_DATA_PATH
 * -model MODEL_DESCRIPTOR_STRING
 * <p/>
 * First verify that the data can be read on your system using the baseline
 * model. Second, find the point in the main method (near the bottom) where a
 * MostFrequentLabelClassifier is constructed. You will be writing new
 * implementations of the ProbabilisticClassifer interface and constructing them
 * there.
 */
public class ProperNameTester {
	static double[][] confusion = new double[5][5];

	public static class ProperNameFeatureExtractor implements
			FeatureExtractor<String, String> {

		/**
		 * This method takes the list of characters representing the proper name
		 * description, and produces a list of features which represent that
		 * description. The basic implementation is that only character-unigram
		 * features are extracted. An easy extension would be to also throw
		 * character bigrams into the feature list, but better features are also
		 * possible.
		 */
		public Counter<String> extractFeatures(String name) {
			name =  "##" + name + "##";
			char[] characters = name.toCharArray();
			Counter<String> features = new Counter<String>();
			// add unigram features
			for (int i = 2; i < characters.length - 2; i++) {
				String uniGram = "" + characters[i];
				features.incrementCount(uniGram, 1.0);
			}
			// add character bigram features
			for (int i = 1; i < characters.length - 2; i++) {
				String biGram = "" + characters[i] + characters[i + 1];
				features.incrementCount(biGram, 1.0);

			}
			// add character trigram features
			for (int i = 1; i < characters.length - 2; i++) {
				String triGram = "" + characters[i] + characters[i + 1]
						+ characters[i + 2];
				;
				features.incrementCount(triGram, 1.0);

			}
			// add capital feature
			if (Character.isUpperCase(name.charAt(2)))
							features.incrementCount("isCaptial", 1);
			if (name.length()<8)
				features.incrementCount("less_four", 1);
			if (name.length()>14)
				features.incrementCount("larger_ten", 1);
			return features;
		}
	}

	private static List<LabeledInstance<String, String>> loadData(
			String fileName) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		List<LabeledInstance<String, String>> labeledInstances = new ArrayList<LabeledInstance<String, String>>();
		while (reader.ready()) {
			String line = reader.readLine();
			String[] parts = line.split("\t");
			String label = parts[0];
			String name = parts[1];
			LabeledInstance<String, String> labeledInstance = new LabeledInstance<String, String>(
					label, name);
			labeledInstances.add(labeledInstance);
		}
		return labeledInstances;
	}

	private static void testClassifier(
			ProbabilisticClassifier<String, String> classifier,
			List<LabeledInstance<String, String>> testData, boolean verbose) {
		double numCorrect = 0.0;
		double numTotal = 0.0;
		double sumConf = 0;
		// initialize confusion
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				confusion[i][j] = 0;

		for (LabeledInstance<String, String> testDatum : testData) {
			String name = testDatum.getInput();
			String label = classifier.getLabel(name);
			String trueLabel = testDatum.getLabel();
			double conf = classifier.getConf();
			sumConf += conf;
			generateConfusion(label, trueLabel);
			double confidence = classifier.getProbabilities(name).getCount(
					label);
			if (label.equals(testDatum.getLabel())) {
				numCorrect += 1.0;
			} else {
				if (verbose) {
					// display an error
					System.err.println("Error: " + name + " guess=" + label
							+ " gold=" + testDatum.getLabel() + " confidence="
							+ confidence);
				}
			}
			numTotal += 1.0;
		}
		double accuracy = numCorrect / numTotal;
		double avgConf = sumConf / numTotal;
		System.out.println("Accuracy: " + accuracy);
		System.out.println("Confidence: " + avgConf);

		for (double a[] : confusion) {
			for (double b : a) {
				System.out.print(b + "\t");
			}
			System.out.println();
		}

	}

	public static void generateConfusion(String label, String trueLabel) {
		
	}

	public static void main(String[] args) throws IOException {
		// Parse command line flags and arguments
		Map<String, String> argMap = CommandLineUtils
				.simpleCommandLineParser(args);

		// Set up default parameters and settings
		String basePath = ".";
		String model = "baseline";
		boolean verbose = false;
		boolean useValidation = true;

		// Update defaults using command line specifications

		// The path to the assignment data
		if (argMap.containsKey("-path")) {
			basePath = argMap.get("-path");
		}
		System.out.println("Using base path: " + basePath);

		// A string descriptor of the model to use
		if (argMap.containsKey("-model")) {
			model = argMap.get("-model");
		}
		System.out.println("Using model: " + model);

		// A string descriptor of the model to use
		if (argMap.containsKey("-test")) {
			String testString = argMap.get("-test");
			if (testString.equalsIgnoreCase("test"))
				useValidation = false;
		}
		System.out.println("Testing on: "
				+ (useValidation ? "validation" : "test"));

		// Whether or not to print the individual speech errors.
		if (argMap.containsKey("-verbose")) {
			verbose = true;
		}

		// Load training, validation, and test data
		List<LabeledInstance<String, String>> trainingData = loadData(basePath
				+ "/pnp-train.txt");
		List<LabeledInstance<String, String>> validationData = loadData(basePath
				+ "/pnp-validate.txt");
		List<LabeledInstance<String, String>> testData = loadData(basePath
				+ "/pnp-test.txt");

		// Learn a classifier
		ProbabilisticClassifier<String, String> classifier = null;
		ProbabilisticClassifierFactory<String, String> factory = new MaximumEntropyClassifier.Factory<String, String, String>(
				1.0, 500, new ProperNameFeatureExtractor());
		classifier = factory.trainClassifier(trainingData);

		// Test classifier
		testClassifier(classifier, (useValidation ? validationData : testData),
				verbose);
	}

	public void train(List<String[]> word_tag) {
		labeledInstances = new ArrayList<LabeledInstance<String, String>>();
		for (String[] pair : word_tag) {
			LabeledInstance<String, String> labeledInstance = new LabeledInstance<String, String>(
					pair[1], pair[0]);
			labeledInstances.add(labeledInstance);
		}
		ProbabilisticClassifierFactory<String, String> factory = new MaximumEntropyClassifier.Factory<String, String, String>(
				1.0, 500, new ProperNameFeatureExtractor());
		classifier = factory.trainClassifier(labeledInstances);
//		classifier = new SuffixTagClassifier.Factory<String, String>()
//				.trainClassifier(labeledInstances);
		System.out.println("Trainging done!!!");
	}

	public String guessUnknown(String word) {
		String tag = classifier.getLabel(word);
		return tag;
	}
	public double getConf() {
		return classifier.getConf();
	}

	List<LabeledInstance<String, String>> labeledInstances;
	ProbabilisticClassifier<String, String> classifier = null;
}
