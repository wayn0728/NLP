package nlp.assignment4;

import java.util.*;
import java.io.*;

import nlp.assignment4.DiscardIBMModeTwoWordAlignmentTester.SentencePair;
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
public class DiscardIBMModeTwoWordAlignmentTester {

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
		void trainModel1(List<SentencePair> trainingSentencePairs);
	}
	  
	static class Indice {
		int I;
		int J;
		CounterMap<Integer,Integer> indiceMap = new CounterMap<Integer,Integer>();
		Indice(int i, int j) {
			I = i;
			J = j;
		}
		public boolean equals(Object o) {
		    if (this == o) return true;
		    if (!(o instanceof Indice)) return false;

		    final Indice indice = (Indice) o;
		    
		    if (I != indice.I) return false;
		    if (J != indice.J) return false;

		    return true;
		  }

	}
	

	static class ModelTwoAligner implements WordAligner {
		Set<String> EnglishVocabulary = new HashSet<String>();
		Set<String> FrenchVocabulary = new HashSet<String>();
		CounterMap<String, String> p = new CounterMap<String, String>();
		CounterMap<Integer, Tri> indiceMap = new CounterMap<Integer, Tri>();
		//double[][][][] indiceArray = new double[80][80][80][80];
		double threshold = 50;
		Set<Integer> ILen = new HashSet();
		Set<Integer> JLen = new HashSet();
		@Override
		public Alignment alignSentencePair(SentencePair sentencePair) {
			Alignment alignment = new Alignment();
			int numFrenchWords = sentencePair.getFrenchWords().size();
			int numEnglishWords = sentencePair.getEnglishWords().size();
			int maxPos = -1;
				for (int fp = 0; fp < numFrenchWords; fp++) {
					String fw = sentencePair.getFrenchWords().get(fp);
					double maxP = 0.2;
					maxPos = 0;
					double sumP = 0;
					for (int ep = 0; ep < numEnglishWords; ep++) {
						String ew = sentencePair.getEnglishWords().get(ep);
						double thisP = p.getCount(ew, fw);
						//double thisIndice = indice.getCount(ep, new Pair(fw, new Pair(numEnglishWords,numFrenchWords)));
						//sumP += thisP * thisIndice;
					}
					for (int ep = 0; ep < numEnglishWords; ep++) {
						String ew = sentencePair.getEnglishWords().get(ep);
						double thisP = p.getCount(ew, fw);
						//double thisIndice = indice.getCount(ep, new Pair(fw, new Pair(numEnglishWords,numFrenchWords)));
						//thisP *= thisIndice;
						thisP /= sumP;
						thisP *= 0.8;
						if (thisP > maxP) {
							maxP = thisP;
							maxPos = ep;
					}
				}
				if (maxPos != -1)
					alignment.addAlignment(maxPos, fp, true);
			}
			return alignment;
		}

		public void trainModel1(List<SentencePair> trainingSentencePair) {
			int maxLen = 0;
			// initialize the uniform probability
			for (SentencePair sp : trainingSentencePair) {
				List<String> el = sp.getEnglishWords();
				List<String> fl = sp.getFrenchWords();
				ILen.add(el.size());
				JLen.add(fl.size());
				if (el.size() > maxLen) 
					maxLen = el.size();
				if (fl.size() > maxLen)
					maxLen = fl.size();
				for (String ew : el)
					EnglishVocabulary.add(ew);
				for (String fw : fl)
					FrenchVocabulary.add(fw);
				for (String ew : el) {
					for (String fw : fl) {
						if (p.containsKey(ew)) {
							Counter<String> map = p.getCounter(ew);
							if (!map.containsKey(fw))
								map.incrementCount(fw, 1.0);
						} else {
							p.incrementCount(ew, fw, 1.0);
						}
					}
				}
			}
			for (int I : ILen) {
				System.out.println(I);
			}
			System.out.println("EnglishVocabulary size = " + EnglishVocabulary.size());
			p.normalize();
			//System.out.println(maxLen);
			//initialize the indice
			for (int I : ILen) {				
				double u = 1.0 / I;
				for (int J : JLen) {
					System.out.println("I = " + I);
					//System.out.println("J = " + J);
					for (int i = 0; i < I; i++) {
						for (int j = 0; j < J; j++) {
							Tri tri = new Tri(j,I,J);
							indiceMap.incrementCount(i, tri, u);							
						}
					}						
				}					
			}
			for (int I : ILen) {				
				double u = 1.0 / I;
				for (int J : JLen) {
					System.out.println("I = " + I);
					//System.out.println("J = " + J);
					for (int i = 0; i < I; i++) {
						for (int j = 0; j < J; j++) {
							//indiceArray[i][j][I][J] = u;	
						}
					}						
				}					
			}
			

			CounterMap<String, String> currentP = new CounterMap<String, String>();
			CounterMap<String, String> lastP = p;
			CounterMap<Integer, Pair> currentIndice = new CounterMap<Integer, Pair>();
			//CounterMap<Integer, Pair> lastIndice = indice;
			int count = 0;
			// TODO judge converge
			double delta = 1000;
			while (delta > threshold) {
				System.out.println(count++);
				
//				//copy from last to current
				currentP = new CounterMap<String, String>();
				currentIndice = new  CounterMap<Integer, Pair>();
//				for (String ew: lastP.keySet()) {
//					Counter<String> lastMap = lastP.getCounter(ew);
//					for (String fw: lastMap.keySet()) {
//						double lastTempP = lastMap.getCount(fw);
//						currentP.incrementCount(ew, fw, lastTempP);
//					}
//				}
								
				for (SentencePair sp : trainingSentencePair) {
					List<String> el = sp.getEnglishWords();
					List<String> fl = sp.getFrenchWords();
					int eLen = el.size();
					int fLen = fl.size();					
					for (int fp = 0; fp < fLen; fp++) {
						double sumP = 0;
						for (int ep = 0; ep < eLen; ep++) {
							double thisP = lastP.getCount(el.get(ep), fl.get(fp));
							Pair len = new Pair(eLen, fLen);
							Pair all = new Pair(fp, len);
							double thisIndice = currentIndice.getCount(ep, all);
							thisP *= thisIndice;
							sumP += thisP;
						}
						for (int ep = 0; ep < eLen; ep++) {
							double thisP = lastP.getCount(el.get(ep), fl.get(fp));
							Pair len = new Pair(eLen, fLen);
							Pair all = new Pair(fp, len);
							double thisIndice = currentIndice.getCount(ep, all);
							thisP *= thisIndice;
							thisP /= sumP;
							currentP.incrementCount(el.get(ep), fl.get(fp), thisP);
						}
					}
				}
				currentP.normalize();
				currentIndice.normalize();
				// cal the delta
				delta = 0;
				for (String ew: currentP.keySet()) {
					Counter<String> lastMap = lastP.getCounter(ew);
					Counter<String> currentMap = currentP.getCounter(ew);
					for (String fw: currentMap.keySet()) {
						double lastTempP = lastMap.getCount(fw);
						double currentTempP = currentMap.getCount(fw);
						delta += Math.abs(lastTempP-currentTempP);		
					}
				}
				System.out.println(delta);
				lastP = currentP;
				//lastIndice = currentIndice;
			}
			p = lastP;
			//indice = lastIndice;
		}

		@Override
		public void train(List<SentencePair> trainingSentencePairs) {
			// TODO Auto-generated method stub
			
		}
		
		public void trainModel2(List<SentencePair> trainingSentencePair) {
			
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
		wordAligner = new ModelTwoAligner();
		wordAligner.trainModel1(trainingSentencePairs);
		//wordAligner.trainModel2(trainingSentencePairs);
		// TODO : build other alignment models

		// Test model
		test(wordAligner, testSentencePairs, testAlignments, verbose);
	}

}
