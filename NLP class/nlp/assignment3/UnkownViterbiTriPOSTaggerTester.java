package nlp.assignment3;

import java.util.*;

import nlp.io.PennTreebankReader;
import nlp.ling.Tree;
import nlp.ling.Trees;
import nlp.util.*;

/**
 * Harness for POS Tagger project.
 */
public class UnkownViterbiTriPOSTaggerTester {

	static final String START_WORD = "<S>";
	static final String STOP_WORD = "</S>";
	static final String START_TAG = "<S>";
	static final String STOP_TAG = "</S>";

	/**
	 * Tagged sentences are a bundling of a list of words and a list of their
	 * tags.
	 */
	static class TaggedSentence {
		List<String> words;
		List<String> tags;

		public int size() {
			return words.size();
		}

		public List<String> getWords() {
			return words;
		}

		public List<String> getTags() {
			return tags;
		}

		public String toString() {
			StringBuilder sb = new StringBuilder();
			for (int position = 0; position < words.size(); position++) {
				String word = words.get(position);
				String tag = tags.get(position);
				sb.append(word);
				sb.append("_");
				sb.append(tag);
			}
			return sb.toString();
		}

		public boolean equals(Object o) {
			if (this == o)
				return true;
			if (!(o instanceof TaggedSentence))
				return false;

			final TaggedSentence taggedSentence = (TaggedSentence) o;

			if (tags != null ? !tags.equals(taggedSentence.tags)
					: taggedSentence.tags != null)
				return false;
			if (words != null ? !words.equals(taggedSentence.words)
					: taggedSentence.words != null)
				return false;

			return true;
		}

		public int hashCode() {
			int result;
			result = (words != null ? words.hashCode() : 0);
			result = 29 * result + (tags != null ? tags.hashCode() : 0);
			return result;
		}

		public TaggedSentence(List<String> words, List<String> tags) {
			this.words = words;
			this.tags = tags;
		}
	}

	/**
	 * States are pairs of tags along with a position index, representing the
	 * two tags preceding that position. So, the START state, which can be
	 * gotten by State.getStartState() is [START, START, 0]. To build an
	 * arbitrary state, for example [DT, NN, 2], use the static factory method
	 * State.buildState("DT", "NN", 2). There isnt' a single final state, since
	 * sentences lengths vary, so State.getEndState(i) takes a parameter for the
	 * length of the sentence.
	 */
	static class State {

		private static transient Interner<State> stateInterner = new Interner<State>(
				new Interner.CanonicalFactory<State>() {
					public State build(State state) {
						return new State(state);
					}
				});

		private static transient State tempState = new State();

		public static State getStartState() {
			return buildState(START_TAG, START_TAG, 0);
		}

		public static State getStopState(int position) {
			return buildState(STOP_TAG, STOP_TAG, position);
		}

		public static State buildState(String previousPreviousTag,
				String previousTag, int position) {
			tempState.setState(previousPreviousTag, previousTag, position);
			return stateInterner.intern(tempState);
		}

		public static List<String> toTagList(List<State> states) {
			List<String> tags = new ArrayList<String>();
			if (states.size() > 0) {
				tags.add(states.get(0).getPreviousPreviousTag());
				for (State state : states) {
					tags.add(state.getPreviousTag());
				}
			}
			return tags;
		}

		public int getPosition() {
			return position;
		}

		public String getPreviousTag() {
			return previousTag;
		}

		public String getPreviousPreviousTag() {
			return previousPreviousTag;
		}

		public State getNextState(String tag) {
			return State.buildState(getPreviousTag(), tag, getPosition() + 1);
		}

		public State getPreviousState(String tag) {
			return State.buildState(tag, getPreviousPreviousTag(),
					getPosition() - 1);
		}

		public boolean equals(Object o) {
			if (this == o)
				return true;
			if (!(o instanceof State))
				return false;

			final State state = (State) o;

			if (position != state.position)
				return false;
			if (previousPreviousTag != null ? !previousPreviousTag
					.equals(state.previousPreviousTag)
					: state.previousPreviousTag != null)
				return false;
			if (previousTag != null ? !previousTag.equals(state.previousTag)
					: state.previousTag != null)
				return false;

			return true;
		}

		public int hashCode() {
			int result;
			result = position;
			result = 29 * result
					+ (previousTag != null ? previousTag.hashCode() : 0);
			result = 29
					* result
					+ (previousPreviousTag != null ? previousPreviousTag
							.hashCode() : 0);
			return result;
		}

		public String toString() {
			return "[" + getPreviousPreviousTag() + ", " + getPreviousTag()
					+ ", " + getPosition() + "]";
		}

		int position;
		String previousTag;
		String previousPreviousTag;

		private void setState(String previousPreviousTag, String previousTag,
				int position) {
			this.previousPreviousTag = previousPreviousTag;
			this.previousTag = previousTag;
			this.position = position;
		}

		private State() {
		}

		private State(State state) {
			setState(state.getPreviousPreviousTag(), state.getPreviousTag(),
					state.getPosition());
		}
	}

	/**
	 * A Trellis is a graph with a start state an an end state, along with
	 * successor and predecessor functions.
	 */
	static class Trellis<S> {
		S startState;
		S endState;
		CounterMap<S, S> forwardTransitions;
		CounterMap<S, S> backwardTransitions;

		/**
		 * Get the unique start state for this trellis.
		 */
		public S getStartState() {
			return startState;
		}

		public void setStartState(S startState) {
			this.startState = startState;
		}

		/**
		 * Get the unique end state for this trellis.
		 */
		public S getEndState() {
			return endState;
		}

		public void setStopState(S endState) {
			this.endState = endState;
		}

		/**
		 * For a given state, returns a counter over what states can be next in
		 * the markov process, along with the cost of that transition. Caution:
		 * a state not in the counter is illegal, and should be considered to
		 * have cost Double.NEGATIVE_INFINITY, but Counters score items they
		 * don't contain as 0.
		 */
		public Counter<S> getForwardTransitions(S state) {
			return forwardTransitions.getCounter(state);

		}

		/**
		 * For a given state, returns a counter over what states can precede it
		 * in the markov process, along with the cost of that transition.
		 */
		public Counter<S> getBackwardTransitions(S state) {
			return backwardTransitions.getCounter(state);
		}

		public void setTransitionCount(S start, S end, double count) {
			forwardTransitions.setCount(start, end, count);
			backwardTransitions.setCount(end, start, count);
		}

		public Trellis() {
			forwardTransitions = new CounterMap<S, S>();
			backwardTransitions = new CounterMap<S, S>();
		}
	}

	/**
	 * A TrellisDecoder takes a Trellis and returns a path through that trellis
	 * in which the first item is trellis.getStartState(), the last is
	 * trellis.getEndState(), and each pair of states is conntected in the
	 * trellis.
	 */
	static interface TrellisDecoder<S> {
		List<S> getBestPath(Trellis<S> trellis);
	}

	static class GreedyDecoder<S> implements TrellisDecoder<S> {
		public List<S> getBestPath(Trellis<S> trellis) {
			List<S> states = new ArrayList<S>();
			S currentState = trellis.getEndState();
			states.add(currentState);
			while (!currentState.equals(trellis.getStartState())) {
				Counter<S> transitions = trellis
						.getBackwardTransitions(currentState);
				S nextState = transitions.argMax();
				states.add(nextState);
				currentState = nextState;
			}
			Collections.reverse(states);
			return states;
		}
	}

	static class POSTagger {

		public LocalTrigramScorer localTrigramScorer;
		TrellisDecoder<State> trellisDecoder;

		// chop up the training instances into local contexts and pass them on
		// to the local scorer.
		public void train(List<TaggedSentence> taggedSentences,
				ProperNameTester pm, ProperNameTester2 pm2) {
			localTrigramScorer.train(
					extractLabeledLocalTrigramContexts(taggedSentences), pm,
					pm2);
		}

		// chop up the validation instances into local contexts and pass them on
		// to the local scorer.
		public void validate(List<TaggedSentence> taggedSentences) {
			localTrigramScorer
					.validate(extractLabeledLocalTrigramContexts(taggedSentences));
		}

		private List<LabeledLocalTrigramContext> extractLabeledLocalTrigramContexts(
				List<TaggedSentence> taggedSentences) {
			List<LabeledLocalTrigramContext> localTrigramContexts = new ArrayList<LabeledLocalTrigramContext>();
			for (TaggedSentence taggedSentence : taggedSentences) {
				localTrigramContexts
						.addAll(extractLabeledLocalTrigramContexts(taggedSentence));
			}
			return localTrigramContexts;
		}

		private List<LabeledLocalTrigramContext> extractLabeledLocalTrigramContexts(
				TaggedSentence taggedSentence) {
			List<LabeledLocalTrigramContext> labeledLocalTrigramContexts = new ArrayList<LabeledLocalTrigramContext>();
			List<String> words = new BoundedList<String>(
					taggedSentence.getWords(), START_WORD, STOP_WORD);
			List<String> tags = new BoundedList<String>(
					taggedSentence.getTags(), START_TAG, STOP_TAG);
			for (int position = 0; position <= taggedSentence.size() + 1; position++) {
				labeledLocalTrigramContexts.add(new LabeledLocalTrigramContext(
						words, position, tags.get(position - 2), tags
								.get(position - 1), tags.get(position)));
			}
			return labeledLocalTrigramContexts;
		}

		/**
		 * Builds a Trellis over a sentence, by starting at the state State, and
		 * advancing through all legal extensions of each state already in the
		 * trellis. You should not have to modify this code (or even read it,
		 * really).
		 */
		private Trellis<State> buildTrellis(List<String> sentence) {
			Trellis<State> trellis = new Trellis<State>();
			trellis.setStartState(State.getStartState());
			State stopState = State.getStopState(sentence.size() + 2);
			trellis.setStopState(stopState);
			Set<State> states = Collections.singleton(State.getStartState());
			trellis.setTransitionCount(State.getStartState(),
					State.getStartState(), 0);
			for (int position = 0; position <= sentence.size() + 1; position++) {
				Set<State> nextStates = new HashSet<State>();
				for (State state : states) {
					if (state.equals(stopState))
						continue;
					LocalTrigramContext localTrigramContext = new LocalTrigramContext(
							sentence, position, state.getPreviousPreviousTag(),
							state.getPreviousTag());
					Counter<String> tagScores = localTrigramScorer
							.getLogScoreCounter(localTrigramContext);
					for (String tag : tagScores.keySet()) {
						State s = trellis.getBackwardTransitions(state)
								.argMax();
						double delta = trellis.getBackwardTransitions(state)
								.getCount(s);
						double score = tagScores.getCount(tag);
						double res = delta + score;
						State nextState = state.getNextState(tag);
						if (!trellis.backwardTransitions.containsKey(nextState))
							trellis.setTransitionCount(state, nextState, res);
						else {
							State s0 = trellis
									.getBackwardTransitions(nextState).argMax();
							double delta0 = trellis.getBackwardTransitions(
									state).getCount(s0);
							if (delta0 > res)
								trellis.setTransitionCount(state, nextState,
										res);
						}
						nextStates.add(nextState);
					}
				}
				// System.out.println("States: "+nextStates);
				states = nextStates;
			}
			return trellis;
		}

		// to tag a sentence: build its trellis and find a path through that
		// trellis
		public List<String> tag(List<String> sentence) {
			Trellis<State> trellis = buildTrellis(sentence);
			List<State> states = trellisDecoder.getBestPath(trellis);
			List<String> tags = State.toTagList(states);
			tags = stripBoundaryTags(tags);
			return tags;
		}

		/**
		 * Scores a tagging for a sentence. Note that a tag sequence not
		 * accepted by the markov process should receive a log score of
		 * Double.NEGATIVE_INFINITY.
		 */
		public double scoreTagging(TaggedSentence taggedSentence) {
			double logScore = 0.0;
			List<LabeledLocalTrigramContext> labeledLocalTrigramContexts = extractLabeledLocalTrigramContexts(taggedSentence);
			for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
				Counter<String> logScoreCounter = localTrigramScorer
						.getLogScoreCounter(labeledLocalTrigramContext);
				String currentTag = labeledLocalTrigramContext.getCurrentTag();
				if (logScoreCounter.containsKey(currentTag)) {
					logScore += logScoreCounter.getCount(currentTag);
				} else {
					logScore += Double.NEGATIVE_INFINITY;
				}
			}
			return logScore;
		}

		private List<String> stripBoundaryTags(List<String> tags) {
			return tags.subList(2, tags.size() - 2);
		}

		public POSTagger(LocalTrigramScorer localTrigramScorer,
				TrellisDecoder<State> trellisDecoder) {
			this.localTrigramScorer = localTrigramScorer;
			this.trellisDecoder = trellisDecoder;
		}
	}

	/**
	 * A LocalTrigramContext is a position in a sentence, along with the
	 * previous two tags -- basically a FeatureVector.
	 */
	static class LocalTrigramContext {
		List<String> words;
		int position;
		String previousTag;
		String previousPreviousTag;

		public List<String> getWords() {
			return words;
		}

		public String getCurrentWord() {
			return words.get(position);
		}

		public String getPreviousWord() {
			return words.get(position - 1);
		}

		public String getPreviousPreviousWord() {
			return words.get(position - 2);
		}

		public int getPosition() {
			return position;
		}

		public String getPreviousTag() {
			return previousTag;
		}

		public String getPreviousPreviousTag() {
			return previousPreviousTag;
		}

		public String toString() {
			return "[" + getPreviousPreviousTag() + ", " + getPreviousTag()
					+ ", " + getCurrentWord() + "]";
		}

		public LocalTrigramContext(List<String> words, int position,
				String previousPreviousTag, String previousTag) {
			this.words = words;
			this.position = position;
			this.previousTag = previousTag;
			this.previousPreviousTag = previousPreviousTag;
		}
	}

	/**
	 * A LabeledLocalTrigramContext is a context plus the correct tag for that
	 * position -- basically a LabeledFeatureVector
	 */
	static class LabeledLocalTrigramContext extends LocalTrigramContext {
		String currentTag;

		public String getCurrentTag() {
			return currentTag;
		}

		public String toString() {
			return "[" + getPreviousPreviousTag() + ", " + getPreviousTag()
					+ ", " + getCurrentWord() + "_" + getCurrentTag() + "]";
		}

		public LabeledLocalTrigramContext(List<String> words, int position,
				String previousPreviousTag, String previousTag,
				String currentTag) {
			super(words, position, previousPreviousTag, previousTag);
			this.currentTag = currentTag;
		}
	}

	/**
	 * LocalTrigramScorers assign scores to tags occuring in specific
	 * LocalTrigramContexts.
	 */
	static interface LocalTrigramScorer {
		/**
		 * The Counter returned should contain log probabilities, meaning if all
		 * values are exponentiated and summed, they should sum to one. For
		 * efficiency, the Counter can contain only the tags which occur in the
		 * given context with non-zero model probability.
		 */
		Counter<String> getLogScoreCounter(
				LocalTrigramContext localTrigramContext);

		void train(List<LabeledLocalTrigramContext> localTrigramContexts,
				ProperNameTester pm, ProperNameTester2 pm2);

		void validate(List<LabeledLocalTrigramContext> localTrigramContexts);
	}

	/**
	 * The MostFrequentTagScorer gives each test word the tag it was seen with
	 * most often in training (or the tag with the most seen word types if the
	 * test word is unseen in training. This scorer actually does a little more
	 * than its name claims -- if constructed with restrictTrigrams = true, it
	 * will forbid illegal tag trigrams, otherwise it makes no use of tag
	 * history information whatsoever.
	 */
	static class MostFrequentTagScorer implements LocalTrigramScorer {

		boolean restrictTrigrams; // if true, assign log score of
									// Double.NEGATIVE_INFINITY to illegal tag
									// trigrams.

		CounterMap<String, String> wordsToTags = new CounterMap<String, String>();
		CounterMap<String, String> tagsToWords = new CounterMap<String, String>();
		Counter<String> unknownWordTags = new Counter<String>();
		Counter<String> uniTag = new Counter<String>();
		Counter<List<String>> biTag = new Counter<List<String>>();
		Counter<List<String>> triTag = new Counter<List<String>>();
		Set<String> seenTagTrigrams = new HashSet<String>();

		public int getHistorySize() {
			return 2;
		}

		public Counter<String> getLogScoreCounter(
				LocalTrigramContext localTrigramContext) {
			int position = localTrigramContext.getPosition();
			String word = localTrigramContext.getWords().get(position);
			String previousTag = localTrigramContext.getPreviousTag();
			String previousPreviousTag = localTrigramContext
					.getPreviousPreviousTag();
			Counter<String> tagCounter = unknownWordTags;
			if (wordsToTags.keySet().contains(word)) {
				tagCounter = wordsToTags.getCounter(word);
			} else {
				int a = 0;
			}
			Counter<String> logScoreCounter = new Counter<String>();
			for (String tag : tagCounter.keySet()) {
				double p_wt = 0;
				Counter<String> wordSet = tagsToWords.getCounter(tag);
				if (wordSet.containsKey(word))
					p_wt = tagsToWords.getCount(tag, word);
				else
					p_wt = unknownWordTags.getCount(tag)
							/ unknownWordTags.totalCount();
				List<String> bi = new ArrayList<String>();
				bi.add(previousPreviousTag);
				bi.add(previousTag);
				List<String> tri = new ArrayList<String>();
				tri.add(previousPreviousTag);
				tri.add(previousTag);
				tri.add(tag);
				double p_gram = 0;
				if (triTag.containsKey(tri)) {
					p_gram = triTag.getCount(tri)
							/ (biTag.getCount(bi) + uniTag.size());
				} else
					p_gram = 1 / (biTag.getCount(bi) + uniTag.size());
				double p = p_wt * p_gram;
				double logScore = Math.log(p);
				logScoreCounter.setCount(tag, logScore);
			}
			return logScoreCounter;
		}

		private Set<String> allowedFollowingTags(Set<String> tags,
				String previousPreviousTag, String previousTag) {
			Set<String> allowedTags = new HashSet<String>();
			for (String tag : tags) {
				String trigramString = makeTrigramString(previousPreviousTag,
						previousTag, tag);
				if (seenTagTrigrams.contains((trigramString))) {
					allowedTags.add(tag);
				}
			}
			return allowedTags;
		}

		private String makeTrigramString(String previousPreviousTag,
				String previousTag, String currentTag) {
			return previousPreviousTag + " " + previousTag + " " + currentTag;
		}

		public void train(
				List<LabeledLocalTrigramContext> labeledLocalTrigramContexts,
				ProperNameTester pm, ProperNameTester2 pm2) {
			List<String[]> word_tag = new ArrayList<String[]>();
			int count = 0;
			// collect word-tag counts
			for (LabeledLocalTrigramContext labeledLocalTrigramContext : labeledLocalTrigramContexts) {
				String word = labeledLocalTrigramContext.getCurrentWord();
				String tag = labeledLocalTrigramContext.getCurrentTag();
				if (!wordsToTags.keySet().contains(word)) {
					// word is currently unknown, so tally its tag in the
					// unknown tag counter
					unknownWordTags.incrementCount(tag, 1.0);
					String[] temp = new String[2];
					temp[0] = word;
					temp[1] = tag;
					word_tag.add(temp);
				}
				wordsToTags.incrementCount(word, tag, 1.0);
				tagsToWords.incrementCount(tag, word, 1.0);
				if ((count % 10) == 0) {

				}
				count++;

				// process word_tag pair

				if (labeledLocalTrigramContext.previousTag.equals(START_TAG))
					uniTag.incrementCount(START_TAG, 1.0);
				if (!labeledLocalTrigramContext.previousPreviousTag
						.equals(STOP_TAG))
					uniTag.incrementCount(tag, 1.0);

				List<String> bi = new ArrayList<String>();
				String preciousTag = labeledLocalTrigramContext
						.getPreviousTag();
				String preciousPreviousTag = labeledLocalTrigramContext
						.getPreviousPreviousTag();
				bi.add(preciousPreviousTag);
				bi.add(preciousTag);
				List<String> tri = new ArrayList<String>();
				tri.add(preciousPreviousTag);
				tri.add(preciousTag);
				tri.add(tag);
				biTag.incrementCount(bi, 1.0);
				triTag.incrementCount(tri, 1.0);
				seenTagTrigrams.add(makeTrigramString(
						labeledLocalTrigramContext.getPreviousPreviousTag(),
						labeledLocalTrigramContext.getPreviousTag(),
						labeledLocalTrigramContext.getCurrentTag()));
			}
			wordsToTags = Counters.conditionalNormalize(wordsToTags);
			tagsToWords = Counters.conditionalNormalize(tagsToWords);
			unknownWordTags = Counters.normalize(unknownWordTags);
			int a = word_tag.size();
			pm.train(word_tag);
			pm2.train(word_tag);
		}

		public void validate(
				List<LabeledLocalTrigramContext> labeledLocalTrigramContexts) {
			// no tuning for this dummy model!
		}

		public MostFrequentTagScorer(boolean restrictTrigrams) {
			this.restrictTrigrams = restrictTrigrams;
		}

	}

	private static List<TaggedSentence> readTaggedSentences(String path,
			int low, int high) {
		Collection<Tree<String>> trees = PennTreebankReader.readTrees(path,
				low, high);
		List<TaggedSentence> taggedSentences = new ArrayList<TaggedSentence>();
		Trees.TreeTransformer<String> treeTransformer = new Trees.EmptyNodeStripper();
		for (Tree<String> tree : trees) {
			tree = treeTransformer.transformTree(tree);
			List<String> words = new BoundedList<String>(new ArrayList<String>(
					tree.getYield()), START_WORD, STOP_WORD);
			List<String> tags = new BoundedList<String>(new ArrayList<String>(
					tree.getPreTerminalYield()), START_TAG, STOP_TAG);
			taggedSentences.add(new TaggedSentence(words, tags));
		}
		return taggedSentences;
	}

	private static double[] evaluateTagger(double thre, double thre2,
			POSTagger posTagger, List<TaggedSentence> taggedSentences,
			Set<String> trainingVocabulary, boolean verbose,
			ProperNameTester pm, ProperNameTester2 pm2) {
		double numTags = 0.0;
		double numTagsCorrect = 0.0;
		double numUnknownWords = 0.0;
		double numUnknownWordsCorrect = 0.0;
		int numDecodingInversions = 0;
		double threshold = thre;
		double threshold2 = thre2;
		for (TaggedSentence taggedSentence : taggedSentences) {
			List<String> words = taggedSentence.getWords();
			List<String> goldTags = taggedSentence.getTags();
			List<String> guessedTags = posTagger.tag(words);
			for (int position = 0; position < words.size() - 1; position++) {
				String word = words.get(position);
				String goldTag = goldTags.get(position);
				String guessedTag = guessedTags.get(position);
				if (!trainingVocabulary.contains(word)) {
					String tempTag = pm.guessUnknown(word);
					String tempTag2 = pm2.guessUnknown(word);
					double conf = pm.getConf();
					double conf2 = pm2.getConf();
					if (conf > threshold)
						guessedTag = tempTag;
					else if (conf2 > threshold2)
						guessedTag = tempTag2;
				}
				if (guessedTag.equals(goldTag))
					numTagsCorrect += 1.0;
				numTags += 1.0;
				if (!trainingVocabulary.contains(word)) {
					if (guessedTag.equals(goldTag))
						numUnknownWordsCorrect += 1.0;
					numUnknownWords += 1.0;
				}
			}
			double scoreOfGoldTagging = posTagger.scoreTagging(taggedSentence);
			double scoreOfGuessedTagging = posTagger
					.scoreTagging(new TaggedSentence(words, guessedTags));
			if (scoreOfGoldTagging > scoreOfGuessedTagging) {
				numDecodingInversions++;
				if (verbose)
					System.out
							.println("WARNING: Decoder suboptimality detected.  Gold tagging has higher score than guessed tagging.");
			}
			if (verbose)
				System.out.println(alignedTaggings(words, goldTags,
						guessedTags, true) + "\n");
		}
		System.out.println((numTagsCorrect / numTags)
				+ "\t"
				+ (numUnknownWordsCorrect / numUnknownWords)
				);
		double[] res = new double[2];
		res[0] = numTagsCorrect / numTags;
		res[1] = numUnknownWordsCorrect / numUnknownWords;
		return res;
	}

	// pretty-print a pair of taggings for a sentence, possibly suppressing the
	// tags which correctly match
	private static String alignedTaggings(List<String> words,
			List<String> goldTags, List<String> guessedTags,
			boolean suppressCorrectTags) {
		StringBuilder goldSB = new StringBuilder("Gold Tags: ");
		StringBuilder guessedSB = new StringBuilder("Guessed Tags: ");
		StringBuilder wordSB = new StringBuilder("Words: ");
		for (int position = 0; position < words.size(); position++) {
			equalizeLengths(wordSB, goldSB, guessedSB);
			String word = words.get(position);
			String gold = goldTags.get(position);
			String guessed = guessedTags.get(position);
			wordSB.append(word);
			if (position < words.size() - 1)
				wordSB.append(' ');
			boolean correct = (gold.equals(guessed));
			if (correct && suppressCorrectTags)
				continue;
			guessedSB.append(guessed);
			goldSB.append(gold);
		}
		return goldSB + "\n" + guessedSB + "\n" + wordSB;
	}

	private static void equalizeLengths(StringBuilder sb1, StringBuilder sb2,
			StringBuilder sb3) {
		int maxLength = sb1.length();
		maxLength = Math.max(maxLength, sb2.length());
		maxLength = Math.max(maxLength, sb3.length());
		ensureLength(sb1, maxLength);
		ensureLength(sb2, maxLength);
		ensureLength(sb3, maxLength);
	}

	private static void ensureLength(StringBuilder sb, int length) {
		while (sb.length() < length) {
			sb.append(' ');
		}
	}

	private static Set<String> extractVocabulary(
			List<TaggedSentence> taggedSentences) {
		Set<String> vocabulary = new HashSet<String>();
		for (TaggedSentence taggedSentence : taggedSentences) {
			List<String> words = taggedSentence.getWords();
			vocabulary.addAll(words);
		}
		return vocabulary;
	}

	public static void main(String[] args) {
		// Parse command line flags and arguments
		Map<String, String> argMap = CommandLineUtils
				.simpleCommandLineParser(args);

		// Set up default parameters and settings
		String basePath = ".";
		boolean verbose = false;
		boolean useValidation = true;

		// Update defaults using command line specifications

		// The path to the assignment data
		if (argMap.containsKey("-path")) {
			basePath = argMap.get("-path");
		}
		System.out.println("Using base path: " + basePath);

		// Whether to use the validation or test set
		if (argMap.containsKey("-test")) {
			String testString = argMap.get("-test");
			if (testString.equalsIgnoreCase("test"))
				useValidation = false;
		}
		System.out.println("Testing on: "
				+ (useValidation ? "validation" : "test"));

		// Whether or not to print the individual errors.
		if (argMap.containsKey("-verbose")) {
			verbose = true;
		}

		// Read in data
		System.out.print("Loading training sentences...");
		List<TaggedSentence> trainTaggedSentences = readTaggedSentences(
				basePath, 200, 2199);
		Set<String> trainingVocabulary = extractVocabulary(trainTaggedSentences);
		System.out.println("done.");
		System.out.print("Loading validation sentences...");
		List<TaggedSentence> validationTaggedSentences = readTaggedSentences(
				basePath, 2200, 2299);
		System.out.println("done.");
		System.out.print("Loading test sentences...");
		List<TaggedSentence> testTaggedSentences = readTaggedSentences(
				basePath, 2300, 2399);
		System.out.println("done.");

		// Construct tagger components
		// TODO : improve on the MostFrequentTagScorer
		LocalTrigramScorer localTrigramScorer = new MostFrequentTagScorer(false);
		// TODO : improve on the GreedyDecoder
		TrellisDecoder<State> trellisDecoder = new GreedyDecoder<State>();
		ProperNameTester pm = new ProperNameTester();
		ProperNameTester2 pm2 = new ProperNameTester2();
		// Train tagger
		POSTagger posTagger = new POSTagger(localTrigramScorer, trellisDecoder);

		
		posTagger.validate(validationTaggedSentences);
		posTagger.train(trainTaggedSentences, pm, pm2);
		double max = 0;
		double maxI = 0;
		double maxJ = 0;
		// Test tagger
//		for (double i = 0.5; i < 1; i += 0.02) {
//			for (double j = 0.5; j < 1; j += 0.05) {
//				System.out.println("i = " + i + '\t' + "j = " + j);
//				double temp = evaluateTagger(i, j, posTagger,
//						validationTaggedSentences, trainingVocabulary, verbose,
//						pm, pm2)[1];
//				if (temp > max) {
//					maxI = i;
//					maxJ = j;
//					max = temp;
//				}
//				System.out.println("Max till now: " + max);
//				System.out.println("maxI = " + maxI + '\t' + "maxJ = " + maxJ);
//				System.out.println();
//			}
//		}
//		posTagger.train(trainTaggedSentences, pm, pm2);
		double temp[] = evaluateTagger(0.56, 0.65, posTagger,
					testTaggedSentences, trainingVocabulary, verbose,
					pm, pm2);
//			pAccSum += temp[0];
//			PUnknownSum += temp[1];
//		System.out.println("Average Tag Accuracy = " + pAccSum/10);
//		System.out.println("Average Unkown Accuracy = " + PUnknownSum/10);
	}
}
