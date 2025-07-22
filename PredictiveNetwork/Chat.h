#include "NetworkCudaTexture.cuh"
#include <vector>

constexpr uint32_t bufferSize = 8u;
constexpr uint32_t maxOutput = 32u;

class Chat {
	std::string inputStr, outputStr;
	std::vector<float> inputData, outputData;
	float locations[bufferSize], characters[bufferSize];
	NetworkCudaTexture<1024u> network;
	
public:
	Chat() {

	}

	void setInput(std::string s) {
		inputStr = s;
		inputData.resize(s.size() * 2u);
		for (uint32_t i = 0u; i < s.size(); i++) {
			inputData[i] = (float)s[i] * (1.0f / 256.0f);
		}
	}

	void setOutput(std::string s) {
		outputStr = s;
		outputData.resize(s.size() * 2u);
		for (uint32_t i = 0u; i < s.size(); i++) {
			outputData[i] = (float)s[i] * (1.0f / 256.0f);
		}
	}

	void updateInput() {
		network.enableCudaValues();
		network.getValues(0u, bufferSize, locations);
#pragma unroll bufferSize
		for (uint32_t i = 0u; i < bufferSize; i++) {
			if (locations[i] >= 0.0f) {
				uint32_t j = (uint32_t)(locations[i] * inputData.size());
				if (j < inputData.size()) characters[i] = inputData[j];
				else characters[i] = 0.0f;
			} else {
				if (locations[i] < -1.0f) characters[i] = 0.0f;
				uint32_t j = (uint32_t)((locations[i] + 1.0f) * outputData.size());
				if (j < outputData.size()) characters[i] = outputData[j];
				else characters[i] = 0.0f;
			}
		}
		network.setValues(bufferSize, bufferSize, characters);
		network.disableCudaValues();
	}

	void updateOutput() {
		network.enableCuda();
		for (uint32_t i = 0u; i < outputData.size(); i += bufferSize) {
			for (uint32_t j = 0u; j < bufferSize && (i + j) < outputData.size(); j++) {
				uint32_t k = i + j;
				locations[j] = ((float)k / (float)outputData.size());
				characters[j] = outputData[k];
			}
			network.setValues(2u * bufferSize, bufferSize, locations);
			network.setValues(3u * bufferSize, bufferSize, characters);
			network.train();
		}
		network.disableCuda();
	}

	std::string getOutput() {
		network.enableCudaValues();
		network.getValues(2u * bufferSize, bufferSize, locations);
		network.getValues(3u * bufferSize, bufferSize, characters);
		network.disableCudaValues();
		for (uint32_t i = 0u; i < bufferSize; i++) {
			if (locations[i] >= 0.0f) {
				uint32_t j = (uint32_t)(locations[i] * outputData.size());
				if (j < maxOutput) {
					if (j >= outputData.size()) { outputData.resize(j + 1u); }
					outputData[j] = characters[i];
				}
			}
		}
		outputStr.resize(outputData.size());
		for (uint32_t i = 0u; i < outputData.size(); i++) {
			outputStr[i] = (char)(outputData[i] * 256.0f);
		}
		return outputStr;
	}

	void train() {
		network.enableCuda();
		network.train();
		network.disableCuda();
		
	}

	void draw() {
		network.draw();
	}

};