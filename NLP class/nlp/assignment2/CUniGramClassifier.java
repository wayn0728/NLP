package nlp.assignment2;

import java.util.HashMap;
import java.util.List;

import nlp.classify.*;
import nlp.util.Counter;

/**
 * Baseline classifier which always chooses the class seen most frequently in
 * training.
 */
public class CUniGramClassifier<I, L> implements ProbabilisticClassifier<I, L> {
	static HashMap<Character, Integer> cMap = new HashMap<Character, Integer>();
	static HashMap<Integer, Character> iMap = new HashMap<Integer, Character>();
	static double [][] cFreq = new double[5][200];
	static double[] labelCount = new double[5];
	static double total = 0;
	static int index = 0;
	public double conf = 0;
	
	Counter<L> labels;
	
	public static class Factory<I, L> implements
			ProbabilisticClassifierFactory<I, L> {
		public ProbabilisticClassifier<I, L> trainClassifier(
				List<LabeledInstance<I, L>> trainingData) {
			Counter<L> labels = new Counter<L>();

			// initial all array
			for (int i = 0; i < 5; i++) {
				labelCount[i] = 0;
				for (int j = 0; j < 200; j++) {
					cFreq[i][j] = 0;
				}
			}

			// calculate character frequency
			for (LabeledInstance<I, L> datum : trainingData) {
				int cIndex = 0;
				;
				labels.incrementCount(datum.getLabel(), 1.0);
				String thisInstance = (String) datum.getInput();
				String thisLabel = (String) datum.getLabel();
				int labelIndex = 0;
				switch (thisLabel) {
				case "place":
					labelIndex = 0;
					break;
				case "movie":
					labelIndex = 1;
					break;
				case "drug":
					labelIndex = 2;
					break;
				case "person":
					labelIndex = 3;
					break;
				case "company":
					labelIndex = 4;
					break;
				}
				for (int i = 0; i < thisInstance.length(); i++) {
					char c = thisInstance.charAt(i);
					if (cMap.containsKey(c))
						cIndex = cMap.get(c);
					else {
						cIndex = index;
						iMap.put(index, c);
						cMap.put(c, index++);
					}
					cFreq[labelIndex][cIndex]++;
					labelCount[labelIndex]++;
					total++;
				}
			}
//			for (int i = 0; i < 5; i++) {
//				for (int j = 0; j < 100; j++) {
//					System.out.print(iMap.get(j)+":"+cFreq[i][j] + "\t\t");
//				}
//				System.out.println();
//			}
//			for (int i = 0; i < 5; i++) {
//				double sum = 0;
//				for (int j = 0; j < 100; j++) {
//					sum += cFreq[i][j]++;
//				}
//				System.out.print(sum + "\t");
//			}
				
			
			
			return new CUniGramClassifier<I, L>(labels);
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
		for (int i =0; i < instace.length(); i++) {
			char c = instace.charAt(i);
			if (cMap.containsKey(c)) {
				int cIndex = cMap.get(c);
//				double f = cFreq[labelIndex][cIndex];
//				double lc = labelCount[labelIndex];
				thisP = (cFreq[labelIndex][cIndex] + 1) / (labelCount[labelIndex]);
			}
			else {
				thisP = 1/( labelCount[labelIndex]);
			}
			p *= thisP;
		}
		p *= (labelCount[labelIndex]/total);
		return p;
	}

	public CUniGramClassifier(Counter<L> labels) {
		// this.labels = labels;
	}

	public CUniGramClassifier(L label) {
		// this.labels = new Counter<L>();
		// labels.incrementCount(label, 1.0);
	}

	@Override
	public double getConf() {
		
		return conf;
	}


}
