package nlp.assignment4;

import java.util.*;
import java.io.*;

import nlp.assignment4.IBMModeOneWordAlignmentTester.Alignment;
import nlp.io.IOUtils;
import nlp.util.*;

/**
 * Harness for testing word-level alignments. The code is hard-wired for the
 * aligment source to be english and the alignment target to be french (recall
 * that's the direction for translating INTO english in the noisy channel
 * model).
 * 
 * Your projects will implement several methods of word-to-word alignment.
 */
public class BiIBMModeOneWordAlignmentTester {

	static final String ENGLISH_EXTENSION = "e";
	static final String FRENCH_EXTENSION = "f";

	/**
	 * A holder for a pair of sentences, each a list of strings. Sentences in
	 * the test sets have integer IDs, as well, which are used to retreive the
	 * gold standard alignments for those sentences.
	 */
	public static class SentencePair {
		int sentenceID;
		String sourceFile;
		List<String> englishWords;
		List<String> frenchWords;

		public int getSentenceID() {
			return sentenceID;
		}

		public String getSourceFile() {
			return sourceFile;
		}

		public List<String> getEnglishWords() {
			return englishWords;
		}

		public List<String> getFrenchWords() {
			return frenchWords;
		}

		public String toString() {
			StringBuilder sb = new StringBuilder();
			for (int englishPosition = 0; englishPosition < englishWords.size(); englishPosition++) {
				String englishWord = englishWords.get(englishPosition);
				sb.append(englishPosition);
				sb.append(":");
				sb.append(englishWord);
				sb.append(" ");
			}
			sb.append("\n");
			for (int frenchPosition = 0; frenchPosition < frenchWords.size(); frenchPosition++) {
				String frenchWord = frenchWords.get(frenchPosition);
				sb.append(frenchPosition);
				sb.append(":");
				sb.append(frenchWord);
				sb.append(" ");
			}
			sb.append("\n");
			return sb.toString();
		}

		public SentencePair(int sentenceID, String sourceFile,
				List<String> englishWords, List<String> frenchWords) {
			this.sentenceID = sentenceID;
			this.sourceFile = sourceFile;
			this.englishWords = englishWords;
			this.frenchWords = frenchWords;
		}
	}

	/**
	 * Alignments serve two purposes, both to indicate your system's guessed
	 * alignment, and to hold the gold standard alignments. Alignments map index
	 * pairs to one of three values, unaligned, possibly aligned, and surely
	 * aligned. Your alignment guesses should only contain sure and unaligned
	 * pairs, but the gold alignments contain possible pairs as well.
	 * 
	 * To build an alignemnt, start with an empty one and use
	 * addAlignment(i,j,true). To display one, use the render method.
	 */
	public static class Alignment {
		Set<Pair<Integer, Integer>> sureAlignments;
		Set<Pair<Integer, Integer>> possibleAlignments;

		public boolean containsSureAlignment(int englishPosition,
				int frenchPosition) {
			return sureAlignments.contains(new Pair<Integer, Integer>(
					englishPosition, frenchPosition));
		}

		public boolean containsPossibleAlignment(int englishPosition,
				int frenchPosition) {
			return possibleAlignments.contains(new Pair<Integer, Integer>(
					englishPosition, frenchPosition));
		}

		public void addAlignment(int englishPosition, int frenchPosition,
				boolean sure) {
			Pair<Integer, Integer> alignment = new Pair<Integer, Integer>(
					englishPosition, frenchPosition);
			if (sure)
				sureAlignments.add(alignment);
			possibleAlignments.add(alignment);
		}

		public Alignment() {
			sureAlignments = new HashSet<Pair<Integer, Integer>>();
			possibleAlignments = new HashSet<Pair<Integer, Integer>>();
		}

		public static String render(Alignment alignment,
				SentencePair sentencePair) {
			return render(alignment, alignment, sentencePair);
		}

		public static String render(Alignment reference, Alignment proposed,
				SentencePair sentencePair) {
			StringBuilder sb = new StringBuilder();
			for (int frenchPosition = 0; frenchPosition < sentencePair
					.getFrenchWords().size(); frenchPosition++) {
				for (int englishPosition = 0; englishPosition < sentencePair
						.getEnglishWords().size(); englishPosition++) {
					boolean sure = reference.containsSureAlignment(
							englishPosition, frenchPosition);
					boolean possible = reference.containsPossibleAlignment(
							englishPosition, frenchPosition);
					char proposedChar = ' ';
					if (proposed.containsSureAlignment(englishPosition,
							frenchPosition))
						proposedChar = '#';
					if (sure) {
						sb.append('[');
						sb.append(proposedChar);
						sb.append(']');
					} else {
						if (possible) {
							sb.append('(');
							sb.append(proposedChar);
							sb.append(')');
						} else {
							sb.append(' ');
							sb.append(proposedChar);
							sb.append(' ');
						}
					}
				}
				sb.append("| ");
				sb.append(sentencePair.getFrenchWords().get(frenchPosition));
				sb.append('\n');
			}
			for (int englishPosition = 0; englishPosition < sentencePair
					.getEnglishWords().size(); englishPosition++) {
				sb.append("---");
			}
			sb.append("'\n");
			boolean printed = true;
			int index = 0;
			while (printed) {
				printed = false;
				StringBuilder lineSB = new StringBuilder();
				for (int englishPosition = 0; englishPosition < sentencePair
						.getEnglishWords().size(); englishPosition++) {
					String englishWord = sentencePair.getEnglishWords().get(
							englishPosition);
					if (englishWord.length() > index) {
						printed = true;
						lineSB.append(' ');
						lineSB.append(englishWord.charAt(index));
						lineSB.append(' ');
					} else {
						lineSB.append("   ");
					}
				}
				index += 1;
				if (printed) {
					sb.append(lineSB);
					sb.append('\n');
				}
			}
			return sb.toString();
		}
	}

	/**
	 * WordAligners have one method: alignSentencePair, which takes a sentence
	 * pair and produces an alignment which specifies an english source for each
	 * french word which is not aligned to "null". Explicit alignment to
	 * position -1 is equivalent to alignment to "null".
	 */
	static interface WordAligner {
		Alignment alignSentencePair(SentencePair sentencePair);

		void train(List<SentencePair> trainingSentencePairs);
	}

	static class MostFrequentAligner implements WordAligner {
		List<SentencePair> trainingSentencePair;
		Counter<String> EnglishCounter = new Counter<String>();
		Counter<String> FrenchCounter = new Counter<String>();
		Counter<Pair<String, String>> EnglishFrenchCounter = new Counter<Pair<String, String>>();
		int count = 0;

		public void train(List<SentencePair> trainingSentencePair) {
			for (SentencePair sp : trainingSentencePair) {
				if (count++ % 1000 == 0)
					System.out.println(count - 1);
				List<String> el = sp.getEnglishWords();
				List<String> fl = sp.getFrenchWords();
				int eSize = el.size();
				int fSize = fl.size();
				for (String ew : el)
					EnglishCounter.incrementCount(ew, 1.0);
				for (String fw : fl)
					FrenchCounter.incrementCount(fw, 1.0);
				for (int i = 0; i < eSize; i++) {
					for (int j = 0; j < fSize; j++) {
						Pair<String, String> pair = new Pair<String, String>(
								el.get(i), fl.get(j));
						EnglishFrenchCounter.incrementCount(pair, 1.0);
					}
				}
			}
		}

		public Alignment alignSentencePair(SentencePair sentencePair) {
			Alignment alignment = new Alignment();
			int numFrenchWords = sentencePair.getFrenchWords().size();
			int numEnglishWords = sentencePair.getEnglishWords().size();
			for (int ep = 0; ep < numEnglishWords; ep++) {
				Counter<Pair<String, String>> pairCounter = new Counter<Pair<String, String>>();
				HashMap<String, Integer> positionMap = new HashMap<String, Integer>();
				for (int fp = 0; fp < numFrenchWords; fp++) {
					String ew = sentencePair.getEnglishWords().get(ep);
					String fw = sentencePair.getFrenchWords().get(fp);
					Pair<String, String> pair = new Pair<String, String>(ew, fw);
					double pairCount = 0;
					double ewCount = 0;
					double fwCount = 0;

					if (EnglishFrenchCounter.containsKey(pair))
						pairCount = EnglishFrenchCounter.getCount(pair);
					else
						pairCount = 1.0;

					if (EnglishCounter.containsKey(ew))
						ewCount = EnglishCounter.getCount(ew);
					else
						ewCount = 1.0;

					if (FrenchCounter.containsKey(fw))
						fwCount = FrenchCounter.getCount(fw);
					else
						fwCount = 1.0;

					double value = pairCount / (ewCount * fwCount);
					pairCounter.incrementCount(pair, value);
					positionMap.put(fw, fp);
				}
				Pair<String, String> pair = pairCounter.argMax();
				String fw = pair.getSecond();
				int fp = positionMap.get(fw);
				alignment.addAlignment(ep, fp, true);
			}
			return alignment;
		}
	}

	static class ModelOneAligner implements WordAligner {

		Set<String> EnglishVocabulary = new HashSet<String>();
		Set<String> FrenchVocabulary = new HashSet<String>();
		CounterMap<String, String> p = new CounterMap<String, String>();
		CounterMap<String, String> p2 = new CounterMap<String, String>();
		double threshold = 0.5;
		double pNull = 0.2;

		@Override
		public Alignment alignSentencePair(SentencePair sentencePair) {
			Set<Pair> s1 = new HashSet<Pair>();		
			Alignment alignment = new Alignment();
			int numFrenchWords = sentencePair.getFrenchWords().size();
			int numEnglishWords = sentencePair.getEnglishWords().size();
			int maxPos = -1;
			for (int ep = 0; ep < numEnglishWords; ep++) {
				String ew = sentencePair.getEnglishWords().get(ep);
				double sumP = 0;
				// cal null
				sumP += p2.getCount("#", ew) * pNull;
				double maxP = 0;
				// cal each
				for (int fp = 0; fp < numFrenchWords; fp++) {
					String fw = sentencePair.getFrenchWords().get(fp);
					double thisP = (p2.getCount(fw, ew) * (1 - pNull) / (numFrenchWords + 1));
					sumP += thisP;
				}
				for (int fp = 0; fp < numFrenchWords; fp++) {
					String fw = sentencePair.getFrenchWords().get(fp);
					double thisP = (p2.getCount(fw, ew) * (1 - pNull) / (numFrenchWords + 1));
					thisP /= sumP;
					if (thisP > maxP) {
						maxP = thisP;
						maxPos = fp;
					}
				}
				//call null
				double nullP = (p2.getCount("#", ew) * pNull) / sumP;
				if (nullP > maxP) {
					maxPos = -1;
				}
				//assign alignment
				if (maxPos != -1)
					s1.add(new Pair(ep, maxPos));					
			}
			
			Set<Pair> s2 = new HashSet<Pair>();	
			alignment = new Alignment();
			numFrenchWords = sentencePair.getFrenchWords().size();
			numEnglishWords = sentencePair.getEnglishWords().size();
			maxPos = -1;
			for (int fp = 0; fp < numFrenchWords; fp++) {
				String fw = sentencePair.getFrenchWords().get(fp);
				if (fw.equals("W")) {
					int a = 1;
				}
				double sumP = 0;
				// cal null
				sumP += p.getCount("#", fw) * pNull;
				double maxP = 0;
				// cal each
				for (int ep = 0; ep < numEnglishWords; ep++) {
					String ew = sentencePair.getEnglishWords().get(ep);
					double thisP = (p.getCount(ew, fw) * (1 - pNull) / (numEnglishWords + 1));
					sumP += thisP;
				}
				for (int ep = 0; ep < numEnglishWords; ep++) {
					String ew = sentencePair.getEnglishWords().get(ep);
					double thisP = (p.getCount(ew, fw) * (1 - pNull) / (numEnglishWords + 1));
					thisP /= sumP;
					if (thisP > maxP) {
						maxP = thisP;
						maxPos = ep;
					}
				}
				//call null
				double nullP = (p.getCount("#", fw) * pNull) / sumP;
				if (nullP > maxP) {
					maxPos = -1;
				}
				//assign alignment
				if (maxPos != -1)
					s2.add(new Pair(maxPos, fp));	
			}
			Set<Pair> s3 = new HashSet<Pair>();	
			for (Pair p: s1) {
				if (s2.contains(p))
					s3.add(p);
			}
			for (Pair p: s3) {
				int ep = (int)p.getFirst();
				int fp = (int)p.getSecond();
				alignment.addAlignment(ep,fp, true);
			}
			
			return alignment;
		}

		@Override
		public void train(List<SentencePair> trainingSentencePair) {
			int test = 0;
			// initialize the uniform probability
			for (SentencePair sp : trainingSentencePair) {
				List<String> el = sp.getEnglishWords();
				// el.add("#");
				List<String> fl = sp.getFrenchWords();
				for (String ew : el)
					EnglishVocabulary.add(ew);
				EnglishVocabulary.add("#");				

				for (String fw : fl)
					FrenchVocabulary.add(fw);
				FrenchVocabulary.add("#");
				
				for (String ew : el) {
					for (String fw : fl) {
						p.incrementCount(ew, fw, 1.0);
						
						p2.incrementCount(fw, ew, 1.0);
					}
				}
				
				for (String ew : el) {
					p2.incrementCount("#", ew, 1.0);
				}
				for (String fw : fl) {
					p.incrementCount("#", fw, 1.0);
				}
			}
			
			System.out.println("EnglishVocabulary size = "
					+ EnglishVocabulary.size());
			p.normalize();
			p2.normalize();

			CounterMap<String, String> currentP = new CounterMap<String, String>();
			CounterMap<String, String> lastP = p;
			int count = 0;
			// TODO judge converge
			double delta = 1000;
			while (delta > threshold) {
				if (count > 10) break;
				System.out.println(count++);
				currentP = new CounterMap<String, String>();

				for (SentencePair sp : trainingSentencePair) {
					List<String> el = sp.getEnglishWords();
					// el.add("#");
					List<String> fl = sp.getFrenchWords();
					for (String fw : fl) {
						double sumP = 0;
						for (String ew : el) {
							sumP += (lastP.getCount(ew, fw) * (1 - pNull) / (el
									.size() + 1));
						}
						sumP += (lastP.getCount("#", fw) * pNull);
						for (String ew : el) {
							double tempP = lastP.getCount(ew, fw) * (1 - pNull)
									/ (el.size() + 1);
							tempP /= sumP;
							currentP.incrementCount(ew, fw, tempP);
						}
						double tempP = lastP.getCount("#", fw);
						tempP /= sumP;
						currentP.incrementCount("#", fw, tempP);
					}
				}
				currentP.normalize();
				// cal the delta
				delta = 0;
				for (String ew : currentP.keySet()) {
					Counter<String> lastMap = lastP.getCounter(ew);
					Counter<String> currentMap = currentP.getCounter(ew);
					for (String fw : currentMap.keySet()) {
						double lastTempP = lastMap.getCount(fw);
						double currentTempP = currentMap.getCount(fw);
						delta += Math.abs(lastTempP - currentTempP);
					}
				}
				System.out.println(delta);
				lastP = currentP;
			}
			p = lastP;
			
			
			
			CounterMap<String, String> currentP2 = new CounterMap<String, String>();
			CounterMap<String, String> lastP2 = p2;
			int count2 = 0;
			// TODO judge converge
			
			while (true) {
				if (count2 > 10) break;
				System.out.println(count2++);
				currentP2 = new CounterMap<String, String>();

				for (SentencePair sp : trainingSentencePair) {
					List<String> el = sp.getEnglishWords();
					// el.add("#");
					List<String> fl = sp.getFrenchWords();
					for (String ew : el) {
						double sumP = 0;
						for (String fw : fl) {
							sumP += (lastP2.getCount(fw, ew) * (1 - pNull) / (fl
									.size() + 1));
						}
						sumP += (lastP2.getCount("#", ew) * pNull);
						for (String fw : fl) {
							double tempP = lastP2.getCount(fw,ew) * (1 - pNull)
									/ (fl.size() + 1);
							tempP /= sumP;
							currentP2.incrementCount(fw, ew, tempP);
						}
						double tempP = lastP2.getCount("#", ew);
						tempP /= sumP;
						currentP2.incrementCount("#", ew, tempP);
					}
				}
				currentP2.normalize();				
				lastP2 = currentP2;
			}
			p2 = lastP2;
		}
	}

	private static void test(WordAligner wordAligner,
			List<SentencePair> testSentencePairs,
			Map<Integer, Alignment> testAlignments, boolean verbose) {
		int proposedSureCount = 0;
		int proposedPossibleCount = 0;
		int sureCount = 0;
		int proposedCount = 0;
		for (SentencePair sentencePair : testSentencePairs) {
			Alignment proposedAlignment = wordAligner
					.alignSentencePair(sentencePair);
			Alignment referenceAlignment = testAlignments.get(sentencePair
					.getSentenceID());
			if (referenceAlignment == null)
				throw new RuntimeException(
						"No reference alignment found for sentenceID "
								+ sentencePair.getSentenceID());
			if (verbose)
				System.out.println("Alignment:\n"
						+ Alignment.render(referenceAlignment,
								proposedAlignment, sentencePair));
			for (int frenchPosition = 0; frenchPosition < sentencePair
					.getFrenchWords().size(); frenchPosition++) {
				for (int englishPosition = 0; englishPosition < sentencePair
						.getEnglishWords().size(); englishPosition++) {
					boolean proposed = proposedAlignment.containsSureAlignment(
							englishPosition, frenchPosition);
					boolean sure = referenceAlignment.containsSureAlignment(
							englishPosition, frenchPosition);
					boolean possible = referenceAlignment
							.containsPossibleAlignment(englishPosition,
									frenchPosition);
					if (proposed && sure)
						proposedSureCount += 1;
					if (proposed && possible)
						proposedPossibleCount += 1;
					if (proposed)
						proposedCount += 1;
					if (sure)
						sureCount += 1;
				}
			}
		}
		System.out.println("Precision: " + proposedPossibleCount
				/ (double) proposedCount);
		System.out.println("Recall: " + proposedSureCount / (double) sureCount);
		System.out.println("AER: "
				+ (1.0 - (proposedSureCount + proposedPossibleCount)
						/ (double) (sureCount + proposedCount)));
	}

	// BELOW HERE IS IO CODE

	private static Map<Integer, Alignment> readAlignments(String fileName) {
		Map<Integer, Alignment> alignments = new HashMap<Integer, Alignment>();
		try {
			BufferedReader in = new BufferedReader(new FileReader(fileName));
			while (in.ready()) {
				String line = in.readLine();
				String[] words = line.split("\\s+");
				if (words.length != 4)
					throw new RuntimeException("Bad alignment file " + fileName
							+ ", bad line was " + line);
				Integer sentenceID = Integer.parseInt(words[0]);
				Integer englishPosition = Integer.parseInt(words[1]) - 1;
				Integer frenchPosition = Integer.parseInt(words[2]) - 1;
				String type = words[3];
				Alignment alignment = alignments.get(sentenceID);
				if (alignment == null) {
					alignment = new Alignment();
					alignments.put(sentenceID, alignment);
				}
				alignment.addAlignment(englishPosition, frenchPosition,
						type.equals("S"));
			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		return alignments;
	}

	private static List<SentencePair> readSentencePairs(String path,
			int maxSentencePairs) {
		List<SentencePair> sentencePairs = new ArrayList<SentencePair>();
		List<String> baseFileNames = getBaseFileNames(path);
		for (String baseFileName : baseFileNames) {
			if (sentencePairs.size() >= maxSentencePairs)
				continue;
			sentencePairs.addAll(readSentencePairs(baseFileName));
		}
		return sentencePairs;
	}

	private static List<SentencePair> readSentencePairs(String baseFileName) {
		List<SentencePair> sentencePairs = new ArrayList<SentencePair>();
		String englishFileName = baseFileName + "." + ENGLISH_EXTENSION;
		String frenchFileName = baseFileName + "." + FRENCH_EXTENSION;
		try {
			BufferedReader englishIn = new BufferedReader(new FileReader(
					englishFileName));
			BufferedReader frenchIn = new BufferedReader(new FileReader(
					frenchFileName));
			while (englishIn.ready() && frenchIn.ready()) {
				String englishLine = englishIn.readLine();
				String frenchLine = frenchIn.readLine();
				Pair<Integer, List<String>> englishSentenceAndID = readSentence(englishLine);
				Pair<Integer, List<String>> frenchSentenceAndID = readSentence(frenchLine);
				if (!englishSentenceAndID.getFirst().equals(
						frenchSentenceAndID.getFirst()))
					throw new RuntimeException("Sentence ID confusion in file "
							+ baseFileName + ", lines were:\n\t" + englishLine
							+ "\n\t" + frenchLine);
				sentencePairs.add(new SentencePair(englishSentenceAndID
						.getFirst(), baseFileName, englishSentenceAndID
						.getSecond(), frenchSentenceAndID.getSecond()));
			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		return sentencePairs;
	}

	private static Pair<Integer, List<String>> readSentence(String line) {
		int id = -1;
		List<String> words = new ArrayList<String>();
		String[] tokens = line.split("\\s+");
		for (int i = 0; i < tokens.length; i++) {
			String token = tokens[i];
			if (token.equals("<s"))
				continue;
			if (token.equals("</s>"))
				continue;
			if (token.startsWith("snum=")) {
				String idString = token.substring(5, token.length() - 1);
				id = Integer.parseInt(idString);
				continue;
			}
			words.add(token.intern());
		}
		return new Pair<Integer, List<String>>(id, words);
	}

	private static List<String> getBaseFileNames(String path) {
		List<File> englishFiles = IOUtils.getFilesUnder(path, new FileFilter() {
			public boolean accept(File pathname) {
				if (pathname.isDirectory())
					return true;
				String name = pathname.getName();
				return name.endsWith(ENGLISH_EXTENSION);
			}
		});
		List<String> baseFileNames = new ArrayList<String>();
		for (File englishFile : englishFiles) {
			String baseFileName = chop(englishFile.getAbsolutePath(), "."
					+ ENGLISH_EXTENSION);
			baseFileNames.add(baseFileName);
		}
		return baseFileNames;
	}

	private static String chop(String name, String extension) {
		if (!name.endsWith(extension))
			return name;
		return name.substring(0, name.length() - extension.length());
	}

	public static void main(String[] args) {
		// Parse command line flags and arguments
		Map<String, String> argMap = CommandLineUtils
				.simpleCommandLineParser(args);

		// Set up default parameters and settings
		String basePath = ".";
		int maxTrainingSentences = 0;
		boolean verbose = false;
		String dataset = "mini";
		String model = "baseline";

		// Update defaults using command line specifications
		if (argMap.containsKey("-path")) {
			basePath = argMap.get("-path");
			System.out.println("Using base path: " + basePath);
		}
		if (argMap.containsKey("-sentences")) {
			maxTrainingSentences = Integer.parseInt(argMap.get("-sentences"));
			System.out.println("Using an additional " + maxTrainingSentences
					+ " training sentences.");
		}
		if (argMap.containsKey("-data")) {
			dataset = argMap.get("-data");
			System.out.println("Running with data: " + dataset);
		} else {
			System.out
					.println("No data set specified.  Use -data [miniTest, validate, test].");
		}
		if (argMap.containsKey("-model")) {
			model = argMap.get("-model");
			System.out.println("Running with model: " + model);
		} else {
			System.out.println("No model specified.  Use -model modelname.");
		}
		if (argMap.containsKey("-verbose")) {
			verbose = true;
		}

		// Read appropriate training and testing sets.
		List<SentencePair> trainingSentencePairs = new ArrayList<SentencePair>();
		if (!dataset.equals("miniTest") && maxTrainingSentences > 0)
			trainingSentencePairs = readSentencePairs(basePath + "/training",
					maxTrainingSentences);
		List<SentencePair> testSentencePairs = new ArrayList<SentencePair>();
		Map<Integer, Alignment> testAlignments = new HashMap<Integer, Alignment>();
		if (dataset.equalsIgnoreCase("test")) {
			testSentencePairs = readSentencePairs(basePath + "/test",
					Integer.MAX_VALUE);
			testAlignments = readAlignments(basePath
					+ "/answers/test.wa.nonullalign");
		} else if (dataset.equalsIgnoreCase("validate")) {
			testSentencePairs = readSentencePairs(basePath + "/trial",
					Integer.MAX_VALUE);
			testAlignments = readAlignments(basePath + "/trial/trial.wa");
		} else if (dataset.equalsIgnoreCase("miniTest")) {
			testSentencePairs = readSentencePairs(basePath + "/mini",
					Integer.MAX_VALUE);
			testAlignments = readAlignments(basePath + "/mini/mini.wa");
		} else {
			throw new RuntimeException("Bad data set mode: " + dataset
					+ ", use test, validate, or miniTest.");
		}
		trainingSentencePairs.addAll(testSentencePairs);

		// Build model
		WordAligner wordAligner = null;
		wordAligner = new ModelOneAligner();
		wordAligner.train(trainingSentencePairs);
		// TODO : build other alignment models

		// Test model
		test(wordAligner, testSentencePairs, testAlignments, verbose);
	}

}
