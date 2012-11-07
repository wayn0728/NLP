package nlp.assignments;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.util.*;
import java.text.NumberFormat;
import java.text.DecimalFormat;

import nlp.langmodel.LanguageModel;
import nlp.util.CommandLineUtils;

/**
 * This is the main harness for assignment 1. To run this harness, use
 * <p/>
 * java edu.berkeley.nlp.assignments.LanguageModelTester -path
 * ASSIGNMENT_DATA_PATH -model MODEL_DESCRIPTOR_STRING
 * <p/>
 * First verify that the data can be read on your system. Second, find the point
 * in the main method (near the bottom) where an EmpiricalUnigramLanguageModel
 * is constructed. You will be writing new implementations of the LanguageModel
 * interface and constructing them there.
 */
public class BiTester {

	// HELPER CLASS FOR THE HARNESS, CAN IGNORE
	static class EditDistance {
		static double INSERT_COST = 1.0;
		static double DELETE_COST = 1.0;
		static double SUBSTITUTE_COST = 1.0;

		private double[][] initialize(double[][] d) {
			for (int i = 0; i < d.length; i++) {
				for (int j = 0; j < d[i].length; j++) {
					d[i][j] = Double.NaN;
				}
			}
			return d;
		}

		public double getDistance(List<? extends Object> firstList,
				List<? extends Object> secondList) {
			double[][] bestDistances = initialize(new double[firstList.size() + 1][secondList
					.size() + 1]);
			return getDistance(firstList, secondList, 0, 0, bestDistances);
		}

		private double getDistance(List<? extends Object> firstList,
				List<? extends Object> secondList, int firstPosition,
				int secondPosition, double[][] bestDistances) {
			if (firstPosition > firstList.size()
					|| secondPosition > secondList.size())
				return Double.POSITIVE_INFINITY;
			if (firstPosition == firstList.size()
					&& secondPosition == secondList.size())
				return 0.0;
			if (Double.isNaN(bestDistances[firstPosition][secondPosition])) {
				double distance = Double.POSITIVE_INFINITY;
				distance = Math.min(
						distance,
						INSERT_COST
								+ getDistance(firstList, secondList,
										firstPosition + 1, secondPosition,
										bestDistances));
				distance = Math.min(
						distance,
						DELETE_COST
								+ getDistance(firstList, secondList,
										firstPosition, secondPosition + 1,
										bestDistances));
				distance = Math.min(
						distance,
						SUBSTITUTE_COST
								+ getDistance(firstList, secondList,
										firstPosition + 1, secondPosition + 1,
										bestDistances));
				if (firstPosition < firstList.size()
						&& secondPosition < secondList.size()) {
					if (firstList.get(firstPosition).equals(
							secondList.get(secondPosition))) {
						distance = Math.min(
								distance,
								getDistance(firstList, secondList,
										firstPosition + 1, secondPosition + 1,
										bestDistances));
					}
				}
				bestDistances[firstPosition][secondPosition] = distance;
			}
			return bestDistances[firstPosition][secondPosition];
		}
	}

	// HELPER CLASS FOR THE HARNESS, CAN IGNORE
	static class SentenceCollection extends AbstractCollection<List<String>> {
		static class SentenceIterator implements Iterator<List<String>> {

			BufferedReader reader;

			public boolean hasNext() {
				try {
					return reader.ready();
				} catch (IOException e) {
					return false;
				}
			}

			public List<String> next() {
				try {
					String line = reader.readLine();
					String[] words = line.split("\\s+");
					List<String> sentence = new ArrayList<String>();
					for (int i = 0; i < words.length; i++) {
						String word = words[i];
						sentence.add(word.toLowerCase());
					}
					return sentence;
				} catch (IOException e) {
					throw new NoSuchElementException();
				}
			}

			public void remove() {
				throw new UnsupportedOperationException();
			}

			public SentenceIterator(BufferedReader reader) {
				this.reader = reader;
			}
		}

		String fileName;

		public Iterator<List<String>> iterator() {
			try {
				BufferedReader reader = new BufferedReader(new FileReader(
						fileName));
				return new SentenceIterator(reader);
			} catch (FileNotFoundException e) {
				throw new RuntimeException("Problem with SentenceIterator for "
						+ fileName);
			}
		}

		public int size() {
			int size = 0;
			Iterator i = iterator();
			while (i.hasNext()) {
				size++;
				i.next();
			}
			return size;
		}

		public SentenceCollection(String fileName) {
			this.fileName = fileName;
		}

		public static class Reader {
			static Collection<List<String>> readSentenceCollection(
					String fileName) {
				return new SentenceCollection(fileName);
			}
		}

	}

	static double calculatePerplexity(LanguageModel languageModel,
			Collection<List<String>> sentenceCollection) {
		double logProbability = 0.0;
		double numSymbols = 0.0;
		double oneProbability = 0;
		for (List<String> sentence : sentenceCollection) {
			oneProbability = languageModel.getSentenceProbability(sentence);
			if (!(oneProbability > 0)) {
				oneProbability = languageModel.getSentenceProbability(sentence);
			}
			logProbability += Math.log(oneProbability) / Math.log(2.0);
			numSymbols += sentence.size();
		}
		double avgLogProbability = logProbability / numSymbols;

		double perplexity = Math.pow(0.5, avgLogProbability);
		return perplexity;
	}

	static double calculateWordErrorRate(LanguageModel languageModel,
			List<SpeechNBestList> speechNBestLists, boolean verbose) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		EditDistance editDistance = new EditDistance();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			List<String> correctSentence = speechNBestList.getCorrectSentence();
			List<String> bestGuess = null;
			double bestScore = Double.NEGATIVE_INFINITY;
			double numWithBestScores = 0.0;
			double distanceForBestScores = 0.0;
			for (List<String> guess : speechNBestList.getNBestSentences()) {
				double score = Math.log(languageModel
						.getSentenceProbability(guess))
						+ (speechNBestList.getAcousticScore(guess) / 16.0);
				double distance = editDistance.getDistance(correctSentence,
						guess);
				if (score == bestScore) {
					numWithBestScores += 1.0;
					distanceForBestScores += distance;
				}
				if (score > bestScore || bestGuess == null) {
					bestScore = score;
					bestGuess = guess;
					distanceForBestScores = distance;
					numWithBestScores = 1.0;
				}
			}
			// double distance = editDistance.getDistance(correctSentence,
			// bestGuess);
			totalDistance += distanceForBestScores / numWithBestScores;
			totalWords += correctSentence.size();
			if (verbose) {
				if (distanceForBestScores > 0.0) {
					System.out.println();
					displayHypothesis("GUESS:", bestGuess, speechNBestList,
							languageModel);
					displayHypothesis("GOLD: ", correctSentence,
							speechNBestList, languageModel);
					// System.out.println("GOLD:  "+correctSentence);
				}
			}
		}
		return totalDistance / totalWords;
	}

	private static NumberFormat nf = new DecimalFormat("0.00E00");

	private static void displayHypothesis(String prefix, List<String> guess,
			SpeechNBestList speechNBestList, LanguageModel languageModel) {
		double acoustic = speechNBestList.getAcousticScore(guess) / 16.0;
		double language = Math.log(languageModel.getSentenceProbability(guess));
		System.out.println(prefix + " AM: " + nf.format(acoustic) + " LM: "
				+ nf.format(language) + " Total: "
				+ nf.format(acoustic + language) + " " + guess);
	}

	static double calculateWordErrorRateLowerBound(
			List<SpeechNBestList> speechNBestLists) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		EditDistance editDistance = new EditDistance();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			List<String> correctSentence = speechNBestList.getCorrectSentence();
			double bestDistance = Double.POSITIVE_INFINITY;
			for (List<String> guess : speechNBestList.getNBestSentences()) {
				double distance = editDistance.getDistance(correctSentence,
						guess);
				if (distance < bestDistance)
					bestDistance = distance;
			}
			totalDistance += bestDistance;
			totalWords += correctSentence.size();
		}
		return totalDistance / totalWords;
	}

	static double calculateWordErrorRateUpperBound(
			List<SpeechNBestList> speechNBestLists) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		EditDistance editDistance = new EditDistance();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			List<String> correctSentence = speechNBestList.getCorrectSentence();
			double worstDistance = Double.NEGATIVE_INFINITY;
			for (List<String> guess : speechNBestList.getNBestSentences()) {
				double distance = editDistance.getDistance(correctSentence,
						guess);
				if (distance > worstDistance)
					worstDistance = distance;
			}
			totalDistance += worstDistance;
			totalWords += correctSentence.size();
		}
		return totalDistance / totalWords;
	}

	static double calculateWordErrorRateRandomChoice(
			List<SpeechNBestList> speechNBestLists) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		EditDistance editDistance = new EditDistance();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			List<String> correctSentence = speechNBestList.getCorrectSentence();
			double sumDistance = 0.0;
			double numGuesses = 0.0;
			for (List<String> guess : speechNBestList.getNBestSentences()) {
				double distance = editDistance.getDistance(correctSentence,
						guess);
				sumDistance += distance;
				numGuesses += 1.0;
			}
			totalDistance += sumDistance / numGuesses;
			totalWords += correctSentence.size();
		}
		return totalDistance / totalWords;
	}

	static Collection<List<String>> extractCorrectSentenceList(
			List<SpeechNBestList> speechNBestLists) {
		Collection<List<String>> correctSentences = new ArrayList<List<String>>();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			correctSentences.add(speechNBestList.getCorrectSentence());
		}
		return correctSentences;
	}

	static Set extractVocabulary(Collection<List<String>> sentenceCollection) {
		Set<String> vocabulary = new HashSet<String>();
		for (List<String> sentence : sentenceCollection) {
			for (String word : sentence) {
				vocabulary.add(word);
			}
		}
		return vocabulary;
	}

	public static void main(String[] args) throws IOException {
		// Parse command line flags and arguments
		Map<String, String> argMap = CommandLineUtils
				.simpleCommandLineParser(args);

		// Set up default parameters and settings
		String basePath = ".";
		String model = "baseline";
		boolean verbose = false;

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

		// Whether or not to print the individual speech errors.
		if (argMap.containsKey("-verbose")) {
			verbose = true;
		}
		if (argMap.containsKey("-quiet")) {
			verbose = false;
		}

		// Read in all the assignment data
		String trainingSentencesFile = "/treebank-sentences-spoken-train.txt";
		String validationSentencesFile = "/treebank-sentences-spoken-validate.txt";
		String testSentencesFile = "/treebank-sentences-spoken-test.txt";
		String speechNBestListsPath = "/wsj_n_bst";
		Collection<List<String>> trainingSentenceCollection = SentenceCollection.Reader
				.readSentenceCollection(basePath + trainingSentencesFile);
		Collection<List<String>> validationSentenceCollection = SentenceCollection.Reader
				.readSentenceCollection(basePath + validationSentencesFile);
		Collection<List<String>> testSentenceCollection = SentenceCollection.Reader
				.readSentenceCollection(basePath + testSentencesFile);
		Set trainingVocabulary = extractVocabulary(trainingSentenceCollection);
		List<SpeechNBestList> speechNBestLists = SpeechNBestList.Reader
				.readSpeechNBestLists(basePath + speechNBestListsPath,
						trainingVocabulary);

		// Build the language model
		LanguageModel languageModel = null;
		if (model.equalsIgnoreCase("baseline")) {
			languageModel = new EmpiricalUnigramLanguageModel(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("bigram")) {
			languageModel = new EmpiricalBigramLanguageModel(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("trigram")) {
			languageModel = new EmpiricalTrigramLanguageModel(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("KN")) {
			languageModel = new KNBigramLanguageModel(
					trainingSentenceCollection);
		} else {
			throw new RuntimeException("Unknown model descriptor: " + model);
		}

		int mode = 1; // 1: test r1,r2 2: test K 3:add-1 4:add-delta
		if (mode == 1) {
			// Evaluate the language model
			File f = new File("2-gram result.txt");
			BufferedWriter output = new BufferedWriter(new FileWriter(f));
			double min = 1;
			double step = 0.1;

			double r1 = 0;
			double r2 = 0;
			double minR1 = 0;
			double minR2 = 0;

			for (r1 = 0; r1 <= 1; r1 += step) {
				r2 = 1 - r1;
				EmpiricalBigramLanguageModel.r1 = r1;
				EmpiricalBigramLanguageModel.r2 = r2;
				double wordErrorRate = calculateWordErrorRate(languageModel,
						speechNBestLists, verbose);
				if (min > wordErrorRate) {
					min = wordErrorRate;
					minR1 = r1;
					minR2 = r2;
				}
				System.out.println("r1: " + r1);
				System.out.println("r2: " + r2);
				System.out.println("HUB Word Error Rate: " + wordErrorRate);
				System.out.println("Min Error Rate till now: " + min);
				// System.out.println("minK=" + minK);

				double wsjPerplexity = calculatePerplexity(languageModel,
						validationSentenceCollection);
				double hubPerplexity = calculatePerplexity(languageModel,
						extractCorrectSentenceList(speechNBestLists));
				System.out.println("WSJ Perplexity:  " + wsjPerplexity);
				System.out.println("HUB Perplexity:  " + hubPerplexity);
				System.out.println();

				BigDecimal big_r1 = new BigDecimal(r1);
				BigDecimal big_r2 = new BigDecimal(r2);
				BigDecimal big_wsjPerplexity = new BigDecimal(wsjPerplexity);
				BigDecimal big_hubPerplexity = new BigDecimal(hubPerplexity);
				BigDecimal big_wordErrorRate = new BigDecimal(wordErrorRate);

				big_r1 = big_r1.setScale(1, BigDecimal.ROUND_HALF_UP);
				big_r2 = big_r2.setScale(1, BigDecimal.ROUND_HALF_UP);
				big_wsjPerplexity = big_wsjPerplexity.setScale(2,
						BigDecimal.ROUND_HALF_UP);
				big_hubPerplexity = big_hubPerplexity.setScale(2,
						BigDecimal.ROUND_HALF_UP);
				big_wordErrorRate = big_wordErrorRate.setScale(4,
						BigDecimal.ROUND_HALF_UP);

				output.write(big_r1 + "\t\t\t" + big_r2 + "\t\t\t"
						+ big_wsjPerplexity + "\t\t\t" + big_hubPerplexity
						+ "\t\t\t" + big_wordErrorRate);
				output.write("\n");
			}
			output.write("\n");
			output.write("min WER:" + min + "\n");
			output.write("minR1:" + "\t" + minR1 + "\n");
			output.write("minR2:" + "\t" + minR2 + "\n");
			output.close();
		} else if (mode == 2) {
			File f = new File("2-gram result-k.txt");
			BufferedWriter output = new BufferedWriter(new FileWriter(f));
			double min = 1;
			double minK = 1;
			double k = 0.1;
			double step = 0.1;
			double minPerp = 10000;
			for (k = 0.1; k < 100; k += 0.5) {
				EmpiricalBigramLanguageModel.k = k;
				double wordErrorRate = calculateWordErrorRate(languageModel,
						speechNBestLists, verbose);
				double wsjPerplexity = calculatePerplexity(languageModel,
						validationSentenceCollection);
				double hubPerplexity = calculatePerplexity(languageModel,
						extractCorrectSentenceList(speechNBestLists));
				if (minPerp > wsjPerplexity) {
					minPerp = wsjPerplexity;
					minK = k;
					min = wordErrorRate;
				}
				System.out.println("k: " + k);
				System.out.println("HUB Word Error Rate: " + wordErrorRate);
				System.out.println("Min Error Rate till now: " + min);
				// System.out.println("minK=" + minK);

				System.out.println("WSJ Perplexity:  " + wsjPerplexity);
				System.out.println("HUB Perplexity:  " + hubPerplexity);
				System.out.println();

				BigDecimal big_k = new BigDecimal(k);
				BigDecimal big_wsjPerplexity = new BigDecimal(wsjPerplexity);
				BigDecimal big_hubPerplexity = new BigDecimal(hubPerplexity);
				BigDecimal big_wordErrorRate = new BigDecimal(wordErrorRate);

				big_k = big_k.setScale(1, BigDecimal.ROUND_HALF_UP);
				big_wsjPerplexity = big_wsjPerplexity.setScale(2,
						BigDecimal.ROUND_HALF_UP);
				big_hubPerplexity = big_hubPerplexity.setScale(2,
						BigDecimal.ROUND_HALF_UP);
				big_wordErrorRate = big_wordErrorRate.setScale(4,
						BigDecimal.ROUND_HALF_UP);

				output.write(big_k + "\t\t\t\t" + big_wsjPerplexity + "\t\t\t"
						+ big_hubPerplexity + "\t\t\t" + big_wordErrorRate);
				output.write("\n");
			}
			output.write("\n");
			output.write("min WER:" + min + "\n");
			output.write("minK:" + "\t" + minK + "\n");
			output.close();
		} else if (mode == 3) {

			// Evaluate the language model
			File f = new File("2-gram result-addone.txt");
			BufferedWriter output = new BufferedWriter(new FileWriter(f));

			double wordErrorRate = calculateWordErrorRate(languageModel,
					speechNBestLists, verbose);

			System.out.println("HUB Word Error Rate: " + wordErrorRate);

			// System.out.println("minK=" + minK);

			double wsjPerplexity = calculatePerplexity(languageModel,
					validationSentenceCollection);
			double hubPerplexity = calculatePerplexity(languageModel,
					extractCorrectSentenceList(speechNBestLists));
			System.out.println("WSJ Perplexity:  " + wsjPerplexity);
			System.out.println("HUB Perplexity:  " + hubPerplexity);
			System.out.println();

			BigDecimal big_wsjPerplexity = new BigDecimal(wsjPerplexity);
			BigDecimal big_hubPerplexity = new BigDecimal(hubPerplexity);
			BigDecimal big_wordErrorRate = new BigDecimal(wordErrorRate);

			big_wsjPerplexity = big_wsjPerplexity.setScale(2,
					BigDecimal.ROUND_HALF_UP);
			big_hubPerplexity = big_hubPerplexity.setScale(2,
					BigDecimal.ROUND_HALF_UP);
			big_wordErrorRate = big_wordErrorRate.setScale(4,
					BigDecimal.ROUND_HALF_UP);

			output.write(big_wsjPerplexity + "\t\t\t" + big_hubPerplexity
					+ "\t\t\t" + big_wordErrorRate);
			output.write("\n");

			output.close();

		}
		else if (mode == 4) {

			// Evaluate the language model
			File f = new File("2-gram result-adddelta.txt");
			BufferedWriter output = new BufferedWriter(new FileWriter(f));
			double min = 10000000;
			double delta = 1;
			double minDelta = 0;
			int count = 0;
			for (; count < 10; delta/= 10, count++) {
				EmpiricalBigramLanguageModel.delta = delta;
				double wordErrorRate = calculateWordErrorRate(languageModel,
						speechNBestLists, verbose);
				double wsjPerplexity = calculatePerplexity(languageModel,
						validationSentenceCollection);
				double hubPerplexity = calculatePerplexity(languageModel,
						extractCorrectSentenceList(speechNBestLists));
				if (min > wsjPerplexity) {
					min = wsjPerplexity;
					minDelta = delta;
				}
				System.out.println("delta: " + delta);
				System.out.println("HUB Word Error Rate: " + wordErrorRate);
				System.out.println("WSJ Perplexity:  " + wsjPerplexity);
				System.out.println("HUB Perplexity:  " + hubPerplexity);
				System.out.println();
				
				BigDecimal big_delta=new BigDecimal(delta); 
				BigDecimal big_wsjPerplexity=new BigDecimal(wsjPerplexity); 
				BigDecimal big_hubPerplexity=new BigDecimal(hubPerplexity); 
				BigDecimal big_wordErrorRate=new BigDecimal(wordErrorRate); 
			    
				big_delta=big_delta.setScale(10, BigDecimal.ROUND_HALF_UP); 
				big_wsjPerplexity=big_wsjPerplexity.setScale(2, BigDecimal.ROUND_HALF_UP); 
				big_hubPerplexity=big_hubPerplexity.setScale(2, BigDecimal.ROUND_HALF_UP); 
				big_wordErrorRate=big_wordErrorRate.setScale(4, BigDecimal.ROUND_HALF_UP); 
				
				output.write(big_delta + "\t\t\t" + big_wsjPerplexity + "\t\t\t" + big_hubPerplexity + "\t\t\t" + big_wordErrorRate);
				output.write("\n");
			}
			output.write("\n");
			output.write("min perp:" + min + "\n");
			output.write("minDelta:" + "\t" + minDelta +"\n");
			output.close();
		
		}
		else {
			EmpiricalBigramLanguageModel.k = 0.1;
			double wordErrorRate = calculateWordErrorRate(languageModel,
					speechNBestLists, verbose);
			double wsjPerplexity = calculatePerplexity(languageModel,
					testSentenceCollection);
			double hubPerplexity = calculatePerplexity(languageModel,
					extractCorrectSentenceList(speechNBestLists));
			System.out.println("HUB Word Error Rate: " + wordErrorRate);
			System.out.println("WSJ Perplexity:  " + wsjPerplexity);
			System.out.println("HUB Perplexity:  " + hubPerplexity);
		}

	}

}
