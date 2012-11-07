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
class EmpiricalBigramLanguageModel implements LanguageModel {
	static final String START = "<S>";
	static final String STOP = "</S>";
	double total = 0.0;
	static double k = 0.01;
	static double r1=0.6;
	static double r2=0.4;
	static double delta = 0;
	

	Counter<String> uniWordCounter = new Counter<String>();
	Counter<List<String>> biWordCounter = new Counter<List<String>>();

	public double getWordProbability(List<String> sentence, int index) {
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
		P = r1 * biP + r2 * uniP;
		return P;
	}
	
	public double getWordProbability1(List<String> sentence, int index) {
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
		double delta = 1;
		double v = uniWordCounter.size();
		biP = (count + delta) / (uniSize  + v * delta);

	
		return biP;
	}
	
	public double getWordProbabilityd(List<String> sentence, int index) {
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
		
		biP = (count + delta) / (uniSize + delta * uniWordCounter.size());

	
		return biP;
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
		for (int index = 1; index < stoppedSentence.size(); index++) {
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

	public EmpiricalBigramLanguageModel(
			Collection<List<String>> sentenceCollection) {
		for (List<String> sentence : sentenceCollection) {
			List<String> newSentence = new ArrayList<String>(sentence);
			newSentence.add(START);
			newSentence.addAll(sentence);
			newSentence.add(STOP);
			List<String> stoppedSentence = new ArrayList<String>(newSentence);
			for (int i = 0; i < stoppedSentence.size() - 1; i++) {
				List<String> word = new ArrayList<String>(2);
				String firstWord = stoppedSentence.get(i);
				String secondWord = stoppedSentence.get(i + 1);
				word.add(firstWord);
				word.add(secondWord);
				uniWordCounter.incrementCount(firstWord, 1.0);
				biWordCounter.incrementCount(word, 1.0);
			}
			uniWordCounter.incrementCount(stoppedSentence.get(stoppedSentence.size()-1), 1.0);
		}
		total = uniWordCounter.totalCount();
	}
}
