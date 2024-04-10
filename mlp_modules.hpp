#ifndef MLP_MODULES_HPP
#define MLP_MODULES_HPP

#include "../AlgebraWithSTL/algebra.hpp"

namespace so {

    class Perceptron {

        public:

            Perceptron() {

            }
    };

    class Network {

        public:

            std::vector<Perceptron const &>
            perceptrons;

        public:

            Network() {
                
            }

            ~Network() {
                
            }

            Network
            & addPerceptron(Perceptron const & pPerceptron) {
                perceptrons.push_back(pPerceptron);
                return *this;
            }
    };
}

#endif // MLP_MODULES_HPP
