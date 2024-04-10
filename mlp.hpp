#ifndef MLP_HPP
#define MLP_HPP

#include <algorithm>
#include <memory>
#include <vector>

typedef std::vector<unsigned int> VS;
typedef std::vector<double> VD;

class Perceptron {

    public:

        unsigned int
        sizeOfInput_,
        sizeOfOutput_;

        double
        eta_,
        * input_,
        * output_,
        * delta_,
        * netsum_,
        ** weights_,
        ** weightsSum_;

    public:

        Perceptron() :
        sizeOfInput_(0),
        sizeOfOutput_(0),
        eta_(0),
        input_(nullptr),
        output_(nullptr),
        delta_(nullptr),
        netsum_(nullptr),
        weights_(nullptr),
        weightsSum_(nullptr) {}        

        Perceptron
        & configureNet(unsigned int const & pSizeOfInput, unsigned int const & pSizeOfOutput) {
            freeMemory();
            sizeOfInput_  = pSizeOfInput;
            sizeOfOutput_ = pSizeOfOutput;
            output_       = new double[sizeOfOutput_];
            delta_        = new double[sizeOfOutput_];
            netsum_       = new double[sizeOfOutput_];
            weights_      = new double*[sizeOfOutput_];
            weightsSum_   = new double*[sizeOfOutput_];
            for (unsigned int neuronID = 0; neuronID < sizeOfOutput_; ++neuronID) {
                weights_[neuronID]    = new double[sizeOfInput_ + 1];
                weightsSum_[neuronID] = new double[sizeOfInput_ + 1];
            }

            return *this;
        }

        Perceptron
        & randomizeWeights(double const & pWeightsMin = -.1, double const & pWeightsMax = +.1) {
            if (!weights_){
                return;
            }
            double
            factor = (pWeightsMax - pWeightsMin) / RAND_MAX;
            for (unsigned int toID = 0; toID < sizeOfOutput_; ++toID) {
                for (unsigned int fromID = 0; fromID < sizeOfInput_ + 1; ++fromID) {
                    weights_[toID][fromID]    = pWeightsMin + factor * rand();
                    weightsSum_[toID][fromID] = 0;
                }
            }

            return *this;
        }

        Perceptron
        & setEta(double const & pEta = +.1) {
            eta_ = pEta;

            return *this;
        }

        Perceptron
        & freeMemory() {
            if (weights_) {
                for (unsigned int i = 0; i < sizeOfOutput_; ++i) {
                    if (weights_) {
                        delete[] weights_[i];
                        weights_[i] = nullptr;
                    }
                }
                delete[] weights_;
                weights_ = nullptr;
            }
            if (weightsSum_) {
                for (unsigned int i = 0; i < sizeOfOutput_; ++i) {
                    if (weightsSum_) {
                        delete[] weightsSum_[i];
                        weightsSum_[i] = nullptr;
                    }
                }
                delete[] weightsSum_;
                weightsSum_ = nullptr;
            }
            if (netsum_) {
                delete[] netsum_;
                netsum_ = nullptr;
            }
            if (delta_) {
                delete[] delta_;
                delta_ = nullptr;
            }
            if (output_) {
                delete[] output_;
                output_ = nullptr;
            }
        }

        ~Perceptron() {
            freeMemory();
        }
};

class MLP {

    public:

        unsigned int
        sizeOfInput_;

        double
        * input_;

        Perceptron
        * layer;

    public:

        MLP(VS const & pLayerSizes) :
        sizeOfInput_(pLayerSizes[0]),
        input_(nullptr),
        layer(new Perceptron[pLayerSizes.size() - 1]) {
            // for (unsigned int layerID = 1; layerID < pLayerSizes.size(); ++layerID) {
            //     layer[layerID - 1].configureNet()
            // }shuffle
        }
        
        MLP(VS const & pLayerSizes, double const & pEta, double const & pWeightsMin, double const & pWeightsMax) :
        sizeOfInput_(pLayerSizes[0]),
        input_(nullptr),
        layer(new Perceptron[pLayerSizes.size() - 1]) {
        }  
};
#endif // MLP_HPP