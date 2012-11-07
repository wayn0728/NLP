package nlp.assignment2;

import java.util.HashMap;
import java.util.List;

import nlp.classify.*;
import nlp.util.Counter;

/**
 * Baseline classifier which always chooses the class seen most frequently in
 * training.
 */
public class CBiGramClassifier<I, L> implements ProbabilisticClassifier<I, L> {
	static HashMap<String, Integer> sMap = new HashMap<String, Integer>();
	static double [][] cFreq = new double[5][10000];
	static double[] labelCount = new double[5];
	static int index = 0;
	public double conf = 0;
	
	
	
	Counter<L> labels;
	
	public static class Factory<I, L> implements
			ProbabilisticClassifierFactory<I, L> {
		public ProbabilisticClassifier<I, L> trainClassifier(
				List<LabeledInstance<I, L>> trainingData) {
			return new CBiGramClassifier<I, L>();
		}
	}

	public Counter<L> getProbabilities(I input) {
		Counter<L> counter = new Counter<L>();
		// counter.incrementAll(labels);
		counter.normalize();
		return counter;
	}

	public L getLabel(I input) {
		String thisInstance = (String) input;
		String label = "";
		double maxP = 0;
		int maxI = 0;
		double sumP = 0;
		for (int i = 0; i < 5; i++) {
			double p = calP(thisInstance, i);
			sumP += p;
			if (p > maxP) {
				maxI = i;
				maxP = p;
			}
		}
		conf = maxP / sumP;
		switch (maxI) {
		case 0:
			label = "place";
			break;
		case 1:
			label = "movie";
			break;
		case 2:
			label = "drug";
			break;
		case 3:
			label = "person";
			break;
		case 4:
			label = "company";
			break;
		}
		return (L)label;
	}

	public double calP(String instace, int labelIndex) {
		double p = 1;
		double thisP = 1;
		String instanceNew = "#" + instace + "$";
		for (int i =0; i < instanceNew.length() - 1; i++) {
			String s = instanceNew.substring(i, i+2);
			if (sMap.containsKey(s)) {
				int sIndex = sMap.get(s);
				thisP = (cFreq[labelIndex][sIndex] + 1) / (index + labelCount[labelIndex]);
			}
			else {
				thisP = 1/(index + labelCount[labelIndex]);
			}
			p *= thisP;
		}
		return p;
	}
	
	public double getConf() {
		
		return conf;
	}

	public CBiGramClassifier() {
		// this.labels = labels;
	}

	public CBiGramClassifier(L label) {
		// this.labels = new Counter<L>();
		// labels.incrementCount(label, 1.0);
	}

	
}
