// SGNN: Synaptic-pruning and Genetic Neural Network 
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
const double LEARNING_RATE = 0.001;
const double PRUNE_THRESHOLD = 0.1;
const double ADD_THRESHOLD = 0.5;
const int NUM_NEURONS = 20; 
const int MAX_ITERATIONS = 200; 
const double GLOBAL_THRESHOLD_ADJUSTMENT_RATE = 0.01; 
const double LONG_TERM_STRENGTHEN = 0.05; 
const double LONG_TERM_INHIBIT = 0.05; 
const int THRESHOLD_FOR_ENHANCE = 3; 
const int THRESHOLD_FOR_INHIBIT = 3; 

// Genetic algorithm parameters
const int POPULATION_SIZE = 100; 
const double MUTATION_RATE = 0.01; 
const int NUM_GENERATIONS = 1000; 

//const string LOSS_LOG_FILE = "loss_log_ms.csv";

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
    vector<pair<vector<double>, vector<double>>> data; 
    double totalLoss; 
    vector<Neuron> bestNeurons; 
    double bestLoss; 

    SGNN(const vector<pair<vector<double>, vector<double>>>& targets) : data(targets), totalLoss(0.0), bestLoss(numeric_limits<double>::max()) {
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
                    double strength = ((double)rand() / RAND_MAX) * 2 - 1; 
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
                    double strength = ((double)rand() / RAND_MAX) * 2 - 1; 
                    neuron.connect(j, strength);
                }
            }
        }
    }

    void setInput(const vector<double>& inputs) {
        for (int i = 0; i < inputs.size() && i < neurons.size(); ++i) {
            neurons[i].lastOutput = inputs[i]; 
        }
    }

    double computeLoss(int sampleIndex) {
        double loss = 0.0;
        const auto& targetOutputs = data[sampleIndex].second; 
        for (int i = 0; i < targetOutputs.size(); ++i) {
            double output = neurons[18 + i].lastOutput; 
            loss += abs(output - targetOutputs[i]); 
        }
        return loss / targetOutputs.size(); 
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
        for (int i = 0; i < data.size(); ++i) {
            double output = neurons[18].lastOutput; 
            double error = output - data[i].second[0]; 
            double errorContribution = (abs(error) < 0.1) ? 1 : 0; 

            for (int j = 0; j < NUM_NEURONS; ++j) {
                for (auto& conn : neurons[j].synapticStrengths) {
                    conn.second -= LEARNING_RATE * errorContribution * neurons[j].lastOutput; 
                }
            }

            neurons[18].updateSynapticStrengths(errorContribution); 
        }

        updateGlobalThreshold(); 
    }

    void updateGlobalThreshold() {
        for (auto& neuron : neurons) {
            neuron.threshold += GLOBAL_THRESHOLD_ADJUSTMENT_RATE * (rand() % 2 - 1); 
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
                    double strength = ((double)rand() / RAND_MAX) * 2 - 1; 
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

	/*void logLoss(int iteration) {
	    ofstream ofs(LOSS_LOG_FILE, ios::app); 
	    if (ofs.is_open()) {
	        ofs << iteration << "," << totalLoss << "\n"; 
	        ofs.close(); 
	    } else {
	        cerr << "Unable to open loss log file for writing." << endl;
	    }
	}*/
	
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
};

void geneticAlgorithm(vector<SGNN>& population) {
    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        for (SGNN& network : population) {
            double totalLoss = 0.0;
            for (int i = 0; i < network.data.size(); ++i) {
                network.setInput(network.data[i].first); 
                network.forwardPropagation();
                totalLoss += network.computeLoss(i); 
            }
            network.totalLoss = totalLoss / network.data.size(); 
        }

        sort(population.begin(), population.end(), [](const SGNN& a, const SGNN& b) {
            return a.totalLoss < b.totalLoss; 
        });

        cout << "Generation " << generation << ", Best Loss: " << population[0].totalLoss << endl;
		//population[0].logLoss(generation);
		
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
	/*ofstream ofs(LOSS_LOG_FILE, ios::out | ios::trunc); 
    if (ofs.is_open()) {
        ofs << "Iteration,Loss\n"; 
        ofs.close();
    } else {
        cerr << "Unable to open loss log file." << endl;
    }*/

    vector<pair<vector<double>, vector<double>>> targets = {
        {{0.4, -0.3, -0.9}, {-0.1, 0.7}}, 
        {{0.2, 0.5, -0.1}, {0.5, -0.3}},
        {{-0.5, 0.5, 0.9}, {0.3, 0.8}},
        {{-0.1, 0.7, -0.4}, {0.7, -0.1}}
    };
    
    SGNN sgnn(targets); 

	//sgnn.loadParameters("parameters_ms.txt");

    double previousLoss = numeric_limits<double>::max(); 
    int noImprovementCount = 0; 
    const int patience = 10; 

    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        for (int i = 0; i < targets.size(); ++i) {
            sgnn.setInput(targets[i].first); 
            sgnn.forwardPropagation(); 
            sgnn.totalLoss = sgnn.computeLoss(i); 
            sgnn.backwardPropagation(); 

            sgnn.pruneSynapticConnections(); 
            sgnn.addSynapticConnections(); 

	        //sgnn.logLoss(iter);
	        
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
        }

        sgnn.outputStates(iter); 
    }

    sgnn.restoreBestParameters();
    cout << "Restored best parameters after initial training." << endl;

    vector<SGNN> population;
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        population.emplace_back(targets); 
        population.back().neurons = sgnn.neurons; 
    }

    geneticAlgorithm(population); 

    sgnn = population[0]; 
    for (const auto& target : targets) {
        sgnn.setInput(target.first); 
        sgnn.forwardPropagation(); 

        cout << "Inputs: ";
        for (const auto& input : target.first) {
            cout << input << " "; 
        }
        cout << "\nOutputs: ";
        for (const auto& output : target.second) {
            cout << output << " "; 
        }
        cout << "\nPredicted Outputs: ";
        for (int i = 0; i < target.second.size(); ++i) {
            cout << sgnn.neurons[18 + i].lastOutput << " "; 
        }
        cout << endl;
    }

    sgnn.saveParameters("parameters_ms.txt"); 
    return 0;
}
