package nlp.assignments;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;

/**
 * A dummy language model -- uses empirical unigram counts, plus a single
 * ficticious count for unknown words.
 */
class KNBigramLanguageModel implements LanguageModel {
	static final String START = "<S>";
	static final String STOP = "</S>";
	double total = 0.0;
	static double D = 0.1;
	double exist = 0;

	HashMap<String, Double> alphaMap = new HashMap<String, Double>();
	HashMap<String, Double> existMap = new HashMap<String, Double>();
	HashMap<String, List<String>> firstContain = new HashMap<String, List<String>>();
	Counter<String> uniWordCounter = new Counter<String>();
	Counter<List<String>> biWordCounter = new Counter<List<String>>();

	public double getWordProbability(List<String> sentence, int index) {
		double uniP = 0;
		double biP = 0;
		double P = 0;

		// cal biP
		List<String> word = new ArrayList<String>(2);
		String firstWord = sentence.get(index - 1);
		String secondWord = sentence.get(index);
		word.add(firstWord);
		word.add(secondWord);
		double count = biWordCounter.getCount(word);
		// double VSize = total+1;
		double uniSize = uniWordCounter.getCount(sentence.get(index - 1));

		if (count != 0) {
			P = (count - D) / uniSize;
		} else {
			double alpha = 1;
			if (uniWordCounter.keySet().contains(firstWord))
				alpha = alphaMap.get(firstWord);
			P = alpha * getUniWordProbability(sentence, index);
		}

		return P;
	}
	
	public double getWordProbability2(List<String> sentence, int index) {
		double uniP = 0;
		double biP = 0;
		double P = 0;

		// cal biP
		List<String> word = new ArrayList<String>(2);
		String firstWord = sentence.get(index - 1);
		String secondWord = sentence.get(index);
		word.add(firstWord);
		word.add(secondWord);
		double count = biWordCounter.getCount(word);
		// double VSize = total+1;
		double uniSize = uniWordCounter.getCount(sentence.get(index - 1));

		if (count != 0) {
			P = (count - D) / uniSize;
		} else {
			double alpha = 1;
			if (uniWordCounter.keySet().contains(firstWord))
				alpha = alphaMap.get(firstWord);
			P = alpha * calUniP2(secondWord);
		}

		return P;
	}

	public double getUniWordProbability(List<String> sentence, int index) {
		String word = sentence.get(index);
		double count = uniWordCounter.getCount(word);
		if (count == 0) {
			// System.out.println("UNKNOWN WORD: "+sentence.get(index));
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
			probability *= getWordProbability2(stoppedSentence, index);
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

	public KNBigramLanguageModel(Collection<List<String>> sentenceCollection) {
		for (List<String> sentence : sentenceCollection) {
			List<String> newSentence = new ArrayList<String>(sentence);
			newSentence.add(START);
			newSentence.addAll(sentence);
			newSentence.add(STOP);
			List<String> stoppedSentence = new ArrayList<String>(newSentence);
			List<String> list = new ArrayList<String>();
			for (int i = 0; i < stoppedSentence.size() - 1; i++) {
				List<String> word = new ArrayList<String>(2);
				String firstWord = stoppedSentence.get(i);
				String secondWord = stoppedSentence.get(i + 1);
				word.add(firstWord);
				word.add(secondWord);
				uniWordCounter.incrementCount(firstWord, 1.0);
				

				if (!firstContain.containsKey(word)) {
					list = new ArrayList<String>();
					list.add(secondWord);
					firstContain.put(firstWord, list);
				} else {
					list = firstContain.get(firstWord);
					list.add(secondWord);
					firstContain.put(firstWord, list);
				}
				
				if (!biWordCounter.keySet().contains(word)) {
					if (!existMap.containsKey(secondWord)) {
						existMap.put(secondWord, Double.valueOf((1)));
					} else {
						Double count = existMap.get(secondWord);
						count++;
						existMap.put(secondWord, count);
					}
					exist++;
				}		
				biWordCounter.incrementCount(word, 1.0);
			}
			uniWordCounter.incrementCount(
					stoppedSentence.get(stoppedSentence.size() - 1), 1.0);
		}
		total = uniWordCounter.totalCount();
		System.out.println(uniWordCounter.keySet().size());
		calAlpha();
	}

	private void calAlpha() {
		Iterator uniItr = uniWordCounter.keySet().iterator();
		int num = 0;
		while (uniItr.hasNext()) {
			System.out.print(num++);
			if (num == 27776) {
				int a = 1;
			}
			String word = (String) uniItr.next();
			if (word.equals("</S>"))
				continue;
			List<String> nonZeroList = firstContain.get(word);
			double p_1 = 0;
			double p_2 = 0;
			for (int i = 0; i < nonZeroList.size(); i++) {
				String uniWord = nonZeroList.get(i);
				p_1 += calBiP(word, uniWord);
				p_2 += calUniP2(uniWord);
			}
			double alpha = (1 - p_1) / (1 - p_2);
			alphaMap.put(word, alpha);
		}
	}

	private double calBiP(String first, String second) {
		List<String> word = new ArrayList<String>(2);
		word.add(first);
		word.add(second);
		double p;
		double count = biWordCounter.getCount(word);
		// double VSize = total+1;
		double uniSize = uniWordCounter.getCount(first);
		p = (count - D) / uniSize;
		return p;
	}
	
	
	private double calUniP(String word) {
		double p;
		double count = uniWordCounter.getCount(word);
		double V = uniWordCounter.size();
		double delta = 1;
		p = (count + delta) / (total + delta * V);
		// double VSize = total+1;
		return p;
	}
	
	private double calUniP2(String word) {
		double p;
		double c = 0; 
		if (existMap.containsKey(word)) {
			c = existMap.get(word);
			p = c / exist;
		} 
		else {
			p = calUniP(word);
		}
		return p;
	}
}
