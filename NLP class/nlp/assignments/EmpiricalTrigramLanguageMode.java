package nlp.assignments;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;

/**
 * A dummy language model -- uses empirical unigram counts, plus a single
 * ficticious count for unknown words.
 */
class EmpiricalTrigramLanguageModel implements LanguageModel {
	static final String START = "<S>";
	static final String STOP = "</S>";
	double total = 0.0;
	static double k = 0.1;
	static public double r1 = 0;
	static public double r2 = 0;
	static public double r3 = 0;
	static public double r4 = 0.8;
	static public double r5 = 0.2;

	Counter<String> uniWordCounter = new Counter<String>();
	Counter<List<String>> biWordCounter = new Counter<List<String>>();
	Counter<List<String>> triWordCounter = new Counter<List<String>>();
	
	public double getWordProbability(List<String> sentence, int index) {
		double triP = 0;
		double uniP = 0;
		double biP = 0;
		double P = 0;

		// cal triP
		List<String> word = new ArrayList<String>(3);
		List<String> biWord = new ArrayList<String>(2);
		
		String firstWord = sentence.get(index - 2);
		String secondWord = sentence.get(index - 1);
		String thirdWord = sentence.get(index);
		
		biWord.add(firstWord);
		biWord.add(secondWord);
		word.add(firstWord);
		word.add(secondWord);
		word.add(thirdWord);
		double count = triWordCounter.getCount(word);
		//double VSize = uniWordCounter.keySet().size();
		double biSize = biWordCounter.getCount(biWord);
		// if (uniSize != 0) uniSize = 1;
		
		triP = (count + k * getBiWordProbability(sentence, index)) / (biSize + k);
		//triP = count  / biSize ;
		
		biP = getBiWordProbability(sentence, index);
		uniP = getUniWordProbability(sentence, index);


		// cal P using uniP and biP
		P = (r1 * triP) + (r2 * biP) + (r3 * uniP);
		return P;
	}

	public double getBiWordProbability(List<String> sentence, int index) {
		double uniP = 0;
		double biP = 0;
		double P = 0;
		
		// cal biP
		List<String> word = new ArrayList<String>(2);
		String firstWord = sentence.get(index-1);
		String secondWord = sentence.get(index);
		word.add(firstWord);
		word.add(secondWord);
		double count = biWordCounter.getCount(word);
		//double VSize = total+1;
		double uniSize = uniWordCounter.getCount(sentence.get(index-1));
		
		biP = (count + k * getUniWordProbability(sentence, index)) / (uniSize + k);

		// cal uniP		
		uniP = getUniWordProbability(sentence, index);
		
		// cal P using uniP and biP
		P = r4 * biP + r5 * uniP;
		return P;
	}

	public double getUniWordProbability(List<String> sentence, int index) {
	    String word = sentence.get(index);
	    double count = uniWordCounter.getCount(word);
	    if (count == 0) {
//	      System.out.println("UNKNOWN WORD: "+sentence.get(index));
	      return 1.0 / (total + 1.0);
	    }
	    return count / (total + 1.0);
	  }
	
	public double getSentenceProbability(List<String> sentence) {
		List<String> newSentence = new ArrayList<String>();
		newSentence.add(START);
		newSentence.addAll(sentence);
		newSentence.add(STOP);
		List<String> stoppedSentence = new ArrayList<String>(newSentence);
		double probability = 1.0;
		for (int index = 2; index < stoppedSentence.size(); index++) {
			probability *= getWordProbability(stoppedSentence, index);
		}
		return probability;

	}

	List<String> generateWord() {
		double sample = Math.random();
		double sum = 0.0;
		for (List<String> word : biWordCounter.keySet()) {
			sum += biWordCounter.getCount(word) / total;
			if (sum > sample) {
				return word;
			}
		}
		List<String> temp = new ArrayList<String>(2);
		temp.add("*UNKNOWN*");
		temp.add("*UNKNOWN*");
		return temp;
	}

	public List<String> generateSentence() {
		List<String> sentence = new ArrayList<String>();
		List<String> word = generateWord();
		while (!word.get(1).equals(STOP)) {
			sentence.addAll(word);
			word = generateWord();
		}
		return sentence;
	}

	public EmpiricalTrigramLanguageModel(
			Collection<List<String>> sentenceCollection) {
		for (List<String> sentence : sentenceCollection) {
			List<String> newSentence = new ArrayList<String>(sentence);
			newSentence.add(START);
			newSentence.add(START);
			newSentence.addAll(sentence);
			newSentence.add(STOP);
			newSentence.add(STOP);
			List<String> stoppedSentence = new ArrayList<String>(newSentence);
			for (int i = 0; i < stoppedSentence.size() - 2; i++) {
				List<String> word;
				String firstWord;
				String secondWord;
				String thirdWord;
				
				if (i != 0 && i != stoppedSentence.size() -1) {
					// build a biword
					word = new ArrayList<String>(2);
					firstWord = stoppedSentence.get(i);
					secondWord = stoppedSentence.get(i + 1);
					word.add(firstWord);
					word.add(secondWord);
					biWordCounter.incrementCount(word, 1.0);
					uniWordCounter.incrementCount(firstWord, 1.0);
				}				

				// build a tri word
				word = new ArrayList<String>(3);
				firstWord = stoppedSentence.get(i);
				secondWord = stoppedSentence.get(i + 1);
				thirdWord = stoppedSentence.get(i + 2);
				word.add(firstWord);
				word.add(secondWord);
				word.add(thirdWord);				
				triWordCounter.incrementCount(word, 1.0);
			}
		}
		total = biWordCounter.totalCount();
	}
}
