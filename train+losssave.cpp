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

// Hyperparameters
const double LEARNING_RATE = 0.01;
const double PRUNE_THRESHOLD = 0.1;
const double ADD_THRESHOLD = 0.5;
const int NUM_NEURONS = 20; // Number of neurons
const int MAX_ITERATIONS = 1000; // Maximum number of iterations
const double GLOBAL_THRESHOLD_ADJUSTMENT_RATE = 0.01; // Global threshold adjustment rate
const double LONG_TERM_STRENGTHEN = 0.05; // Long term enhancement amplitude
const double LONG_TERM_INHIBIT = 0.05; // The magnitude of long-term inhibition
const int THRESHOLD_FOR_ENHANCE = 3; // Enhanced threshold
const int THRESHOLD_FOR_INHIBIT = 3; // Threshold of inhibition

// Genetic algorithm parameters
const int POPULATION_SIZE = 100; // Population size
const double MUTATION_RATE = 0.1; // Variation rate
const int NUM_GENERATIONS = 10; // The number of generations

const string LOSS_LOG_FILE = "loss_log.csv";

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
    vector<double> targetOutputs;
    double totalLoss;
    vector<Neuron> bestNeurons;
    double bestLoss;

    SGNN(const vector<double>& targets) : targetOutputs(targets), totalLoss(0.0), bestLoss(numeric_limits<double>::max()) {
        neurons.resize(NUM_NEURONS);
        bestNeurons.resize(NUM_NEURONS);
        srand(static_cast<unsigned int>(time(0)));

        initializeParameters();
    }

    void initializeParameters() {
        for (int i = 0; i < NUM_NEURONS; ++i) {
            neurons[i].threshold = (rand() % 10) * 0.1;
            for (int j = 0; j < NUM_NEURONS; ++j) {
                if (i != j && rand() % 2 == 0) {
                    double strength = ((double)rand() / RAND_MAX) * 2 - 1; // [-1, 1]
                    neurons[i].connect(j, strength);
                }
            }
        }
    }

    void resetParameters() {
        for (auto& neuron : neurons) {
            neuron.lastOutput = 0.0;
            neuron.threshold = (rand() % 10) * 0.1;
            neuron.synapticStrengths.clear();
            neuron.enhanceCount = 0;
            neuron.inhibitCount = 0;
            for (int j = 0; j < NUM_NEURONS; ++j) {
                if (&neuron != &neurons[j] && rand() % 2 == 0) {
                    double strength = ((double)rand() / RAND_MAX) * 2 - 1; // [-1, 1]
                    neuron.connect(j, strength);
                }
            }
        }
    }

    void setInput(const vector<double>& inputs) {
        for (int i = 0; i < inputs.size() && i < 3; ++i) { // Only process the first 3 neurons
            neurons[i].lastOutput = inputs[i]; // Output directly set as input value
        }
    }

    double computeLoss() {
        double loss = 0.0;
        for (int i = 0; i < targetOutputs.size(); ++i) {
            double output = neurons[18 + i].lastOutput; // Obtain the final outputs of neurons 18 and 19
            loss += abs(output - targetOutputs[i]);
        }
        return loss / targetOutputs.size(); // MAE
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

    void backwardPropagation() {
        vector<double> errors;
        for (int i = 0; i < targetOutputs.size(); ++i) {
            double output = neurons[18 + i].lastOutput;
            double error = output - targetOutputs[i];
            errors.push_back(error);
        }

        for (int i = 0; i < NUM_NEURONS; ++i) {
            for (auto& conn : neurons[i].synapticStrengths) {
                double errorContribution = (i >= 18) ? errors[i - 18] : 0.0;
                conn.second -= LEARNING_RATE * errorContribution * neurons[i].lastOutput; 
            }
        }

        for (int i = 0; i < targetOutputs.size(); ++i) {
            bool success = (abs(errors[i]) < 0.1);
            neurons[18 + i].updateSynapticStrengths(success);
        }

        updateGlobalThreshold(errors);
    }

    void updateGlobalThreshold(const vector<double>& errors) {
        double globalError = 0.0;
        for (double error : errors) {
            globalError += abs(error);
        }
        globalError /= errors.size();

        for (auto& neuron : neurons) {
            neuron.threshold -= GLOBAL_THRESHOLD_ADJUSTMENT_RATE * globalError;
            neuron.threshold = max(neuron.threshold, -5.0);
            neuron.threshold = min(neuron.threshold, 5.0);
        }
    }

    void pruneSynapticConnections() {
        for (auto& neuron : neurons) {
            neuron.pruneSynapses();
        }
    }

    void addSynapticConnections() {
        for (int i = 0; i < NUM_NEURONS; ++i) {
            for (int j = 0; j < NUM_NEURONS; ++j) {
                if (i != j && rand() % 2 == 0 && neurons[i].synapticStrengths.size() < NUM_NEURONS / 2) {
                    double strength = ((double)rand() / RAND_MAX) * 2 - 1; // [-1, 1]
                    neurons[i].connect(j, strength);
                }
            }
        }
    }

    void outputStates(int iteration) {
        cout << "Iteration: " << iteration << ", Loss: " << totalLoss << endl;
    }

    void saveBestParameters() {
        bestNeurons.clear();
        for (const auto& neuron : neurons) {
            bestNeurons.push_back(neuron.clone());
        }
        bestLoss = totalLoss;
    }

    void restoreBestParameters() {
        totalLoss = bestLoss;
        neurons = bestNeurons;
    }

	void logLoss(int iteration) {
	    ofstream ofs(LOSS_LOG_FILE, ios::app);
	    if (ofs.is_open()) {
	        ofs << iteration << "," << totalLoss << "\n";
	        ofs.close();
	    } else {
	        cerr << "Unable to open loss log file for writing." << endl;
	    }
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
            neurons.resize(numNeurons);
            bestNeurons.resize(numNeurons);
            for (auto& neuron : neurons) {
                neuron.loadParameters(ifs);
            }
            ifs.close();
        } else {
            cerr << "Unable to open file for loading parameters." << endl;
        }
    }

    void crossover(const SGNN& other) {
        for (size_t i = 0; i < neurons.size(); ++i) {
            if (rand() % 2 == 0) {
                neurons[i].threshold = other.neurons[i].threshold;
            }
            for (size_t j = 0; j < neurons[i].synapticStrengths.size(); ++j) {
                if (rand() % 2 == 0) {
                    neurons[i].synapticStrengths[j].second = other.neurons[i].synapticStrengths[j].second;
                }
            }
        }
    }

    void mutate() {
        for (auto& neuron : neurons) {
            if (rand() / (double)RAND_MAX < MUTATION_RATE) {
                neuron.threshold += ((double)rand() / RAND_MAX * 2 - 1) * 0.1;
            }
            for (auto& conn : neuron.synapticStrengths) {
                if (rand() / (double)RAND_MAX < MUTATION_RATE) {
                    conn.second += ((double)rand() / RAND_MAX * 2 - 1) * 0.1;
                }
            }
        }
    }
};

void geneticAlgorithm(vector<SGNN>& population, const vector<double>& targets) {
    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        for (SGNN& network : population) {
            network.setInput({0.4, -0.3, -0.9});
            network.forwardPropagation();
            network.totalLoss = network.computeLoss();
        }

        sort(population.begin(), population.end(), [](const SGNN& a, const SGNN& b) {
            return a.totalLoss < b.totalLoss;
        });

        cout << "Generation " << generation << ", Best Loss: " << population[0].totalLoss << endl;

        vector<SGNN> newPopulation;
        for (int i = 0; i < POPULATION_SIZE; ++i) {
            SGNN parent1 = population[i];
            SGNN parent2 = population[rand() % (POPULATION_SIZE / 2)];
            
            parent1.crossover(parent2);
            parent1.mutate();
            
            newPopulation.push_back(parent1);
        }

        population = newPopulation;
    }
}

int main() {
    ofstream ofs(LOSS_LOG_FILE, ios::out | ios::trunc);
        if (ofs.is_open()) {
            ofs << "Iteration,Loss\n";
            ofs.close();
        } else {
            cerr << "Unable to open loss log file." << endl;
    }
    vector<double> targets = {-0.1, 0.7};
    SGNN sgnn(targets);

    sgnn.loadParameters("parameters.txt");

    vector<double> inputs = {0.4, -0.3, -0.9};

    double previousLoss = numeric_limits<double>::max();
    int noImprovementCount = 0;
    const int patience = 10;

    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        sgnn.setInput(inputs);
        sgnn.forwardPropagation();
        sgnn.totalLoss = sgnn.computeLoss();
        sgnn.backwardPropagation();

        sgnn.pruneSynapticConnections();
        sgnn.addSynapticConnections();

        sgnn.logLoss(iter);

        if (sgnn.totalLoss > previousLoss) {
            noImprovementCount++;
            if (noImprovementCount >= patience) {
                cout << "Detected overfitting, resetting parameters..." << endl;
                sgnn.resetParameters();
                noImprovementCount = 0;
            }
        } else {
            noImprovementCount = 0;
            if (sgnn.totalLoss < sgnn.bestLoss) {
                sgnn.saveBestParameters();
            }
        }
        
		previousLoss = sgnn.totalLoss;

        sgnn.outputStates(iter);
    }

    sgnn.restoreBestParameters();
    cout << "Restored best parameters after initial training." << endl;

    vector<SGNN> population;
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        population.emplace_back(targets);
        population.back().neurons = sgnn.neurons;
    }

    geneticAlgorithm(population, targets);

    sgnn = population[0];
    sgnn.setInput(inputs);
    sgnn.forwardPropagation();
    cout << "Final outputs for the best network after genetic algorithm:" << endl;
    cout << "Neuron 18 Output: " << sgnn.neurons[18].lastOutput << endl;
    cout << "Neuron 19 Output: " << sgnn.neurons[19].lastOutput << endl;

    sgnn.saveParameters("parameters.txt");
    return 0;
}

