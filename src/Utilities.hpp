#pragma once
#include "Network.hpp"

template <typename T>
void shuffle(std::vector<T>& data, const int seed) {
	std::mt19937 rng(seed);
	std::shuffle(data.begin(), data.end(), rng);
}


std::pair<LabelledSet, LabelledSet> splitData(const LabelledSet& train, const int trainCount, const int seed) {
	assert(trainCount < train.size());


	std::vector<int> idx;
	idx.reserve(train.size());
	for (int i = 0; i < train.size(); i++) {
		idx.push_back(i);
	}

	shuffle(idx, seed);

	LabelledSet validation;
	LabelledSet trainOut;

	validation.reserve(trainCount);
	for (int i = 0; i < trainCount; i++) {
		validation.push_back(train[idx[i]]);
	}

	trainOut.reserve(train.size() - trainCount);
	for (int i = trainCount; i < train.size(); i++) {
		trainOut.push_back(train[idx[i]]);
	}


	return { trainOut, validation };
}