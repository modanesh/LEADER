#ifndef RANDOM_STREAMS_H
#define RANDOM_STREAMS_H

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <despot/util/random.h>

namespace despot {

/**
 * A RandomStreams object represents multiple random number sequences, where each
 * entry is independently and identically drawn from [0, 1].
 */
class RandomStreams {
private:
	mutable int position_;

public:
	  std::vector<std::vector<double> > streams_; // streams_[i] is associated with i-th particle

	/**
	 * Constructs multiple random sequences of the same length.
	 *
	 * @param num_streams number of sequences
	 * @param length sequence length
	 */
	RandomStreams(int num_streams, int length);
	RandomStreams(){;}
	/**
	 * Returns the number of sequences.
	 */
	int NumStreams() const;

	/**
	 * Returns the length of the sequences.
	 */
	int Length() const;

	void Advance() const;
	void Back() const;

	void position(int value) const;
	int position() const;

	bool Exhausted() const;

	double Entry(int stream) const;
	double Entry(int stream, int position) const;

	friend std::ostream& operator<<(std::ostream& os, const RandomStreams& stream);
	void ImportStream(std::istream& in,int num_streams, int length);

	RandomStreams(const RandomStreams& src);
};

} // namespace despot

#endif
