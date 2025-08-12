#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <algorithm>
#include <map>
#include <limits>
#include <fstream>

using namespace std;

const double PRUNE_THRESHOLD = 0.1;
const double ADD_THRESHOLD = 0.5;
int NUM_NEURONS = 20;
const double GLOBAL_THRESHOLD_ADJUSTMENT_RATE = 0.01;
const double LONG_TERM_STRENGTHEN = 0.05;
const double LONG_TERM_INHIBIT = 0.05;
const int THRESHOLD_FOR_ENHANCE = 3;
const int THRESHOLD_FOR_INHIBIT = 3;

class Neuron {
public:
    double lastOutput;
    double threshold;
    vector<pair<int, double>> synapticStrengths;
    int enhanceCount;
    int inhibitCount;

    Neuron() : lastOutput(0.0), threshold(0.0), enhanceCount(0), inhibitCount(0) {}

    double activationFunction(double input) const {
        return (1 - exp(-input + threshold)) / (1 + exp(-input + threshold));
    }

    void connect(int neuronIndex, double strength) {
        synapticStrengths.emplace_back(neuronIndex, strength);
    }

    void pruneSynapses() {
        synapticStrengths.erase(remove_if(synapticStrengths.begin(), synapticStrengths.end(),
            [](const pair<int, double>& conn) { return abs(conn.second) < PRUNE_THRESHOLD; }), synapticStrengths.end());
    }

    Neuron clone() const {
        return *this;
    }

    void updateSynapticStrengths(bool success) {
        if (success) {
            enhanceCount++;
            inhibitCount = 0;
            if (enhanceCount >= THRESHOLD_FOR_ENHANCE) {
                for (auto& conn : synapticStrengths) {
                    conn.second += LONG_TERM_STRENGTHEN;
                }
                enhanceCount = 0;
            }
        } else {
            inhibitCount++;
            enhanceCount = 0;
            if (inhibitCount >= THRESHOLD_FOR_INHIBIT) {
                for (auto& conn : synapticStrengths) {
                    conn.second -= LONG_TERM_INHIBIT;
                }
                inhibitCount = 0;
            }
        }
    }

    void saveParameters(ofstream& ofs) const {
        ofs << threshold << " " << lastOutput << " " << enhanceCount << " " << inhibitCount << " " << synapticStrengths.size() << " ";
        for (const auto& conn : synapticStrengths) {
            ofs << conn.first << " " << conn.second << " ";
        }
    }
    void loadParameters(ifstream& ifs) {
        ifs >> threshold >> lastOutput >> enhanceCount >> inhibitCount;
        int numConnections;
        ifs >> numConnections;
        synapticStrengths.clear();
        for (int i = 0; i < numConnections; ++i) {
            int neuronIndex;
            double strength;
            ifs >> neuronIndex >> strength;
            connect(neuronIndex, strength);
        }
    }
};

class SGNN {
public:
    vector<Neuron> neurons;

    void setInput(const vector<double>& inputs) {
        for (int i = 0; i < inputs.size() && i < 3; ++i) {
            neurons[i].lastOutput = inputs[i];
        }
    }

    void forwardPropagation() {
	    vector<double> inputs(NUM_NEURONS, 0.0);
	    for (int i = 0; i < neurons.size(); ++i) {
	    	neurons[i].lastOutput = 0;
	        for (const auto& conn : neurons[i].synapticStrengths) {
	            int connectedNeuronIndex = conn.first;
	            double strength = conn.second;
	            inputs[i] += neurons[connectedNeuronIndex].lastOutput * strength;
	        }
	        
	        double activation = neurons[i].activationFunction(inputs[i]);
	        neurons[i].lastOutput += activation;
	    }
	}

    void outputStates() {
        for (int i = 0; i < neurons.size(); ++i) {
            cout << "Neuron " << i << " - Last Output: " << neurons[i].lastOutput 
                 << ", Threshold: " << neurons[i].threshold 
                 << ", Connections: ";
            for (const auto& conn : neurons[i].synapticStrengths) {
                cout << "(Neuron " << conn.first << ", Strength: " << conn.second << ") ";
            }
            cout << endl;
        }
        cout << endl;
    }

    void saveParameters(const string& filename) {
        ofstream ofs(filename);
        if (ofs.is_open()) {
            ofs << NUM_NEURONS << endl;
            for (const auto& neuron : neurons) {
                neuron.saveParameters(ofs);
                ofs << endl;
            }
            ofs.close();
        } else {
            cerr << "Unable to open file for saving parameters." << endl;
        }
    }

    void loadParameters(const string& filename) {
	    ifstream ifs(filename);
	    if (ifs.is_open()) {
	        int numNeurons;
	        ifs >> numNeurons;
	        NUM_NEURONS = numNeurons;
	        neurons.resize(numNeurons);
	        for (auto& neuron : neurons) {
	            neuron.loadParameters(ifs);
	        }
	        ifs.close();
	    } else {
	        cerr << "Unable to open file for loading parameters." << endl;
	    }
	}

};

int main() {
    SGNN sgnn;
    sgnn.loadParameters("parameters.txt");
	//sgnn.outputStates();
    vector<double> inputs = {0.2, 0.5, -0.1};
    sgnn.setInput(inputs);
    sgnn.forwardPropagation();
    cout << "Neuron 18 Output: " << sgnn.neurons[18].lastOutput << endl;
    cout << "Neuron 19 Output: " << sgnn.neurons[19].lastOutput << endl;
    sgnn.outputStates();
    return 0;
}

