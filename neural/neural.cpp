#include "Network.hpp"
#include "MNISTLoader.hpp"
#include <cuda_runtime.h>
#include "TestPrint.cuh"
//Split test set into a validation and trainingSet
//Returns: a pair {trainingSet, validationSet}
std::pair<LabelledSet, LabelledSet> createValidationSet(const LabelledSet& train, const int validationCount) {
	//For now, just take the first validationCount elements, later - randomise
	assert(validationCount < train.size());

	LabelledSet validation;
	LabelledSet trainOut;

	validation.reserve(validationCount);
	for (int i = 0; i < validationCount; i++) {
		validation.push_back(train[i]);
	}

	trainOut.reserve(train.size() - validationCount);
	for (int i = validationCount; i < train.size(); i++) {
		trainOut.push_back(train[i]);
	}


	return { trainOut, validation };
}

int main() {

	const auto [train, validation] = createValidationSet(readLabelledData("C:\\Users\\Sam\\Downloads\\train-images.idx3-ubyte", "C:\\Users\\Sam\\Downloads\\train-labels.idx1-ubyte"), 1000);
	
	for (int i = 0; i < 10; i++) {
		print(train[i]);
	}


	for (int i = 0; i < 10; i++) {
		print(validation[i]);
	}

	
	const auto test = readLabelledData("C:\\Users\\Sam\\Downloads\\t10k-images.idx3-ubyte", "C:\\Users\\Sam\\Downloads\\t10k-labels.idx1-ubyte");

	GPUTrain(train, { 28 * 28, 30, 10 });


	return 0;
	const std::filesystem::path jsonPath = "C:/Users/Sam/Desktop/mynetwork.json";

	std::cout << "Train or test? ";
	std::string in;
	std::cin >> in;
	std::transform(in.begin(), in.end(), in.begin(),
		[](unsigned char c) { return std::tolower(c); });


	if (in == "train") {
		for (int i = 0; i < 10; i++) {
			print(train[i]);
		}

		const Hyperparameters params = {
	.hiddenLayers = {30},
	.learningRate = 3.0f,
	.activationFunction = ActivationFunctionType::Sigmoid,
	.epochs = 30,
	.batchSize = 10
		};

		Network network;
		network.train(train, params, 30, [&]() -> void {
			int validation_correct = 0;
			static int epoch = 1;

			
			const auto crossEntropy = [](const Eigen::VectorXf& y,
				 Eigen::VectorXf a) -> Eigen::VectorXf {
					const float eps = 1e-6f;  //Map to [eps, 1-eps] for sigmoid activation (if eps ~= 0 or eps ~= 1 then log produces NaN).
					a = a.array().min(1.0f - eps).max(eps);

					const Eigen::VectorXf ones = Eigen::VectorXf::Ones(y.size());

					Eigen::VectorXf term1 = y.array() * a.array().log();
					Eigen::VectorXf term2 = (ones - y).array() * (ones - a).array().log();

					return -1.0f * (term1 + term2);
				};

			float cost_validation = 0.0f;

			for (const auto& [x, y] : validation) {
				const auto yhat = network.predict(x);
				
				cost_validation += crossEntropy(y, yhat).sum();
			}

			cost_validation /= (float)validation.size();

			float cost_train = 0.0f;
			for (const auto& [x, y] : train) {
				const auto yhat = network.predict(x);
				

				cost_train += crossEntropy(y, yhat).sum();
			}
			cost_train /= (float)train.size();

			std::cout << cost_validation << "," << cost_train << "\n";
			/*
			std::cout << std::format("Epoch {}/{}. {}/{} correct on the validation set ({}%) and {}/{} correct on a random sample from the training set ({}%).\n",
				epoch++, params.epochs,
				validation_correct, validation.size(),
				static_cast<float>(validation_correct) * 100.0f / static_cast<float>(validation.size()),
				train_correct, trainSampleSize,
				static_cast<float>(train_correct) * 100.0f / static_cast<float>(trainSampleSize));*/
				
			});
		
		network.WriteNetworkAsJson(jsonPath);
	}
	else if (in == "test") {
		Network network(jsonPath);
		while (true) {
			std::cout << std::format("\nEnter number in range [{}, {}]: ", 0, test.size() - 1);
			int idx;
			std::cin >> idx;
			assert(idx >= 0 && idx < test.size() );
			print(test[idx]);
			const auto pred = network.predict(test[idx].first);
			Eigen::Index maxidx;
			pred.maxCoeff(&maxidx);
			std::cout << std::format("Predicted {}.", (int)maxidx);
		}

	}
	else {
		return -1;
	}
	return 0;
} 


