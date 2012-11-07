package nlp.assignment3;

import java.util.HashMap;
import java.util.List;

import nlp.classify.*;
import nlp.util.Counter;
import nlp.util.CounterMap;

/**
 * Baseline classifier which always chooses the class seen most frequently in
 * training.
 */
public class SuffixTagClassifier<I, L> implements ProbabilisticClassifier<I, L> {
	static HashMap<String, Integer> sMap = new HashMap<String, Integer>();
	static double[][] cFreq = new double[5][10000];
	static double[] labelCount = new double[5];
	static int index = 0;
	public double conf = 0;
	static Counter<String> tagMap = new Counter<String>();
	static CounterMap<String, String> suffixMap = new CounterMap<String, String>();
	static Counter<String> suffixNumMap = new Counter<String>();
	static double theta;
	Counter<L> labels;
	static int m = 4;
	public static class Factory<I, L> implements
			ProbabilisticClassifierFactory<I, L> {

		@Override
		public ProbabilisticClassifier<I, L> trainClassifier(
				List<LabeledInstance<I, L>> trainingData) {
			for (LabeledInstance<I, L> datum: trainingData) {
				String tag = (String)datum.getLabel();
				String word = (String)datum.getInput();
				tagMap.incrementCount(tag, 1);
				int pos = 0;
				if (word.length() >= m)
					pos = word.length()-m;
				for (int i = pos; i < word.length(); i++) {
					suffixMap.incrementCount(tag, word.substring(i), 1);
					suffixNumMap.incrementCount(word.substring(i), 1);
				}
			}			
			double p_a = 1.0 / tagMap.size();
			//System.out.println(p_a);
			double sum = 0.0;
			for (String thisTag: tagMap.keySet()) {
				double p = tagMap.getCount(thisTag) / tagMap.totalCount();
				//System.out.println(p);
				sum += ((p-p_a) * (p-p_a));
			}
			theta = sum / (tagMap.totalCount()-1);
			//System.out.println(theta);
			return new SuffixTagClassifier(new Counter<String>());			
		}
	}

	public Counter<L> getProbabilities(I input) {
		Counter<L> counter = new Counter<L>();
		// counter.incrementAll(labels);
		counter.normalize();
		return counter;
	}

	public L getLabel(I input) {
		String word = (String)input;
		Counter<String> pMap = new Counter<String>();
		for (String tag: tagMap.keySet()) {
			double p_tag = tagMap.getCount(tag)/ tagMap.totalCount();
			int pos = 0;
			if (word.length() > m) 
				pos = word.length()-m;
			double p = calP(tag, pos, word);
			pMap.incrementCount(tag, p*p_tag);
		}
		conf = pMap.getCount(pMap.argMax()) / pMap.totalCount();
		return (L) pMap.argMax();
	}
	
	private double calP(String tag, int pos, String word) {
		double pActual = calActualP(tag, pos, word);
		if (pos == word.length() -1)
			return pActual;
		else {
			double pBackOff = calActualP(tag, pos + 1, word);
			double p = (pActual + theta * pBackOff) / (1 + theta);
			return p;
		}		
	}
	
	private double calActualP(String tag, int pos, String word) {
		String suffix = word.substring(pos);
		double count = suffixMap.getCount(tag, suffix);
		int length = m;
		if(word.length() < m)
			length = word.length();
		double countSum = suffixMap.getCounter(tag).totalCount()/length;
		double p = count / countSum;
		return p;
	}

	
	public double getConf() {

		return conf;
	}

	public SuffixTagClassifier(Counter<L> labels) {
		// this.labels = labels;
	}

	public SuffixTagClassifier() {
		// this.labels = new Counter<L>();
		// labels.incrementCount(label, 1.0);
	}

}
