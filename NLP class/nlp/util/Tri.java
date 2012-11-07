package nlp.util;

import java.io.Serializable;
import java.util.Comparator;

import com.sun.org.apache.bcel.internal.classfile.Code;


/**
 * A generic-typed pair of objects.
 */
public class Tri<F,S, T> implements Serializable {
  F first;
  S second;
  T third;

  public static class LexicographicPairComparator<F,S,T>  implements Comparator<Tri<F,S,T>> {
    Comparator<F> firstComparator;
    Comparator<S> secondComparator;
    Comparator<T> thirdComparator;

    public int compare(Tri<F, S, T> pair1, Tri<F, S, T> pair2) {
      int firstCompare = firstComparator.compare(pair1.getFirst(), pair2.getFirst());
      if (firstCompare != 0)
        return firstCompare;
      int secondCompare = secondComparator.compare(pair1.getSecond(), pair2.getSecond());
      if (secondCompare != 0)
    	  return secondCompare;      
      return thirdComparator.compare(pair1.getThird(), pair2.getThird());
    }

    public LexicographicPairComparator(Comparator<F> firstComparator, Comparator<S> secondComparator, Comparator<T> thirdComparator) {
      this.firstComparator = firstComparator;
      this.secondComparator = secondComparator;
      this.thirdComparator = thirdComparator;
    }
  }


  public F getFirst() {
    return first;
  }

  public S getSecond() {
    return second;
  }
  
  public T getThird() {
	    return third;
	  }

  public void setFirst(F pFirst) {
    first = pFirst;
  }

  public void setSecond(S pSecond) {
    second = pSecond;
  }
  
  public void setThird(T pThird) {
	    third = pThird;
	  }


  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof Tri)) return false;

    final Tri pair = (Tri) o;

    if (first != null ? !first.equals(pair.first) : pair.first != null) return false;
    if (second != null ? !second.equals(pair.second) : pair.second != null) return false;
    if (third != null ? !third.equals(pair.third) : pair.third != null) return false;

    return true;
  }

  public int hashCode() {
    int result;
    result = (first != null ? first.hashCode() : 0);
    result = 29 * result + (second != null ? second.hashCode() : 0);
    return result;
  }

  public String toString() {
    return "(" + getFirst() + ", " + getSecond() + ")";
  }

  public Tri(F first, S second, T third) {
    this.first = first;
    this.second = second;
    this.third = third;
  }
  
  /**
   * Convenience method for construction of a <code>Pair</code> with
   * the type inference on the arguments. So for instance we can type  
   *     <code>Pair<Tree<String>, Double> treeDoublePair = makePair(tree, count);</code>
   *  instead of,
   *   	 <code>Pair<Tree<String>, Double> treeDoublePair = new Pair<Tree<String>, Double>(tree, count);</code>
   * @author Aria Haghighi
   * @param <F>
   * @param <S>
   * @param f
   * @param s
   * @return <code>Pair<F,S></code> with the arguments <code>f</code>  and <code>s</code>
   */
  public static <F,S,T> Tri<F,S,T> makePair(F f, S s, T t) {
	  return new Tri<F,S, T>(f,s,t);
  }
}

