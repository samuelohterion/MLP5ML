#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "../AlgebraWithSTL/algebra.hpp"
#include <functional>

using namespace alg;

// class ActivationFunction {

//     public:

//         ActivationFunction(STR const &pID) :
//         ID(pID) {

//         }

//         virtual
//         ~ActivationFunction() = 0;

//     public:

//         STR const 
//         ID;

//         D
//         fun,
//         dfun;
// };

// class LossFunction {

//     public:

//         LossFunction(STR const &pID) :
//         ID(pID) {

//         }

//         virtual
//         ~LossFunction() = 0;

//     public:

//         STR const 
//         ID;

//         D
//         fun,
//         dfun;
// };


class Network {

    typedef std::function<void(VD::const_iterator, VD::const_iterator, VD::iterator)> ACTIVATIONFUNCTION;
    typedef long int ADDR;
    
    private:

        Network
        &presentInput(VD::const_iterator pInputBegin) {
            auto
            iIt = i.begin();
            while (iIt < i.end() - 1) {
                *iIt++ = *pInputBegin++;
            }
            return *this;
        }

        Network
        &presentInput(VD const &pInput) {
            return presentInput(pInput.cbegin());
        }

        static void 
        dot(VD & pC, MD const & pA, VD const & pB) {
            D s;
            for (SIZE row = 0; row < pC.size(); ++row) {
                s = 0;
                for (SIZE col = 0; col < pB.size(); ++col) {
                    s += pA[row][col] * pB[col];
                }
                pC[row] = s;
            }        
        }

        static void 
        assignMultipliedDifference(VD &pC, VD const &pA, VD const &pB) {
            for (SIZE row = 0; row < pC.size(); ++row) {
                pC[row] *= (pA[row] - pB[row]);
            }        
        }

        static void 
        assignMultipliedDifference(VD::iterator pCBegin, VD::iterator const &pCEnd, VD::const_iterator pA, VD::const_iterator pB) {
            
            while(pCBegin < pCEnd) {
                *pCBegin++ *= (*pA++ - *pB++);
            }
        }

        static void 
        assignMultipliedDotProduct(VD & pC, VD const & pA, MD const & pB) {
            D s;
            for (SIZE col = 0; col < pC.size(); ++col) {
                s = 0;
                for (SIZE row = 0; row < pB.size(); ++row) {
                    s += pA[row] * pB[row][col];
                }
                pC[col] *= s;
            }        
        }

        Network
        & teach__(SIZE pLayerID) {

            while (0 < pLayerID) {
                --pLayerID;
                dact[pLayerID](o[pLayerID].cbegin(), o[pLayerID].cend() - 1, d[pLayerID].begin());
                // d[layerID] *= d[layerID + 1] | w[layerID + 1];
                assignMultipliedDotProduct(d[pLayerID], d[pLayerID + 1], w[pLayerID + 1]);
            }

            D const
            qm = 1. / (1. - pow(adamBeta1, step)),
            qv = 1. / (1. - pow(adamBeta2, step));

            if (useAdam) {
                MD
                dLdW = d[pLayerID] ^ i,
                dLdWSqr = dLdW * dLdW;
                
                adamM[pLayerID] = (adamBeta1 * adamM[pLayerID]) + (1. - adamBeta1) * dLdW;
                adamV[pLayerID] = (adamBeta2 * adamV[pLayerID]) + (1. - adamBeta2) * dLdWSqr;

                MD
                am = qm * adamM[pLayerID],
                av = qv * adamV[pLayerID];

                for( auto & avIt : av) {
                    for( auto & avValIt : avIt) {
                        avValIt = 1. / (sqrt(avValIt) + 1.e-8);
                    }
                }
                w[pLayerID] -= (eta * am) * av;
            } else {
                w[pLayerID] -= eta * d[pLayerID] ^ i;
            }

            while (++pLayerID < w.size()) {
                if (useAdam) {
                    MD
                    dLdW = d[pLayerID] ^ o[pLayerID - 1],
                    dLdWSqr = dLdW * dLdW;
                    
                    adamM[pLayerID] = (adamBeta1 * adamM[pLayerID]) + (1. - adamBeta1) * dLdW;
                    adamV[pLayerID] = (adamBeta2 * adamV[pLayerID]) + (1. - adamBeta2) * dLdWSqr;

                    MD
                    am = qm * adamM[pLayerID],
                    av = qv * adamV[pLayerID];

                    for( auto & avIt : av) {
                        for( auto & avValIt : avIt) {
                            avValIt = 1. / (sqrt(avValIt) + 1.e-8);
                        }
                    }
                    
                    w[pLayerID] -= (eta * am) * av;
                } else {
                    w[pLayerID] -= eta * d[pLayerID] ^ o[pLayerID - 1];
                }
            }

            return *this;
        }

    public:

        ACTIVATIONFUNCTION const
        actSigmoid = [](VD::const_iterator pCNetBegin, VD::const_iterator pCNetEnd, VD::iterator pOutBegin) {

             std::transform(pCNetBegin, pCNetEnd, pOutBegin, [](D const &pX){return 1. / (1. + exp(-pX));});
        };

        ACTIVATIONFUNCTION const
        dActSigmoid = [](VD::const_iterator pCOutBegin, VD::const_iterator pCOutEnd, VD::iterator pDstBegin) {

            std::transform(pCOutBegin, pCOutEnd, pDstBegin, [](D const &pX){return .001 + pX * (1. - pX);});
        };

        ACTIVATIONFUNCTION const
        actTanh = [](VD::const_iterator pCNetBegin, VD::const_iterator pCNetEnd, VD::iterator pOutBegin) {

             std::transform(pCNetBegin, pCNetEnd, pOutBegin, [](D const &pX){return tanh(pX);});
        };

        ACTIVATIONFUNCTION const
        dActTanh = [](VD::const_iterator pCOutBegin, VD::const_iterator pCOutEnd, VD::iterator pDstBegin) {

            std::transform(pCOutBegin, pCOutEnd, pDstBegin, [](D const &pX){return 1. - pX * pX;});
        };

        ACTIVATIONFUNCTION const
        actReLU = [](VD::const_iterator pCNetBegin, VD::const_iterator pCNetEnd, VD::iterator pOutBegin) {

             std::transform(pCNetBegin, pCNetEnd, pOutBegin, [](D const &pX){return pX < 0 ? .01 * pX : pX;});
        };

        ACTIVATIONFUNCTION const
        dActReLU = [](VD::const_iterator pCOutBegin, VD::const_iterator pCOutEnd, VD::iterator pDstBegin) {

            std::transform(pCOutBegin, pCOutEnd, pDstBegin, [](D const &pX){return pX < 0 ? .01 : 1.;});
        };

        ACTIVATIONFUNCTION const
        actSoftmax = [](VD::const_iterator pCNetBegin, VD::const_iterator pCNetEnd, VD::iterator pOutBegin) {

            D
            maxPred = *std::max_element(pCNetBegin, pCNetEnd);
            std::transform(pCNetBegin, pCNetEnd, pOutBegin, [maxPred](D const &pX){return exp(pX - maxPred);});
            D
            s = 0;
            auto
            outIt = pOutBegin;
            for(auto dummy = pCNetBegin; dummy < pCNetEnd; ++dummy) {
                s += *outIt++;
            }
            s = 1. / s;
            std::transform(pOutBegin, pOutBegin + (pCNetEnd - pCNetBegin), pOutBegin, [maxPred, s](D const &pX){return pX * s;});            
        };

        // ACTIVATIONFUNCTION const
        // dActSoftmax = [](VD::const_iterator pCOutBegin, VD::const_iterator pCOutEnd, VD::iterator pDstBegin) {
        // not neccesary, because softmax is only used in the output layer and the computation of delta includes
        // the derivative of softmax
        //     //std::transform(pCOutBegin, pCOutEnd, pDstBegin, [](D const &pX){return pX < 0 ? .001 : 1.;});
        // };

    private:

        bool
        useAsNetwork;

    public:

        D
        eta;

        bool
        useAdam;

        SIZE
        step;

        D
        adamBeta1,
        adamBeta2;

        Vec<ACTIVATIONFUNCTION>
        act,
        dact;

    private:

        VD
        i;

        MD
        d,
        n,
        o;

        TD
        w;

        TD
        adamM,
        adamV;

    public:

        /*
        Args:
            pLayerSizes:
                sizes of all layers inclusive input pseudo layer. e.g. {2,3,3,2} -> 2 inputs, 2 x 3 hidden, 2 outputs
            pActivationFunctionIDs:
                vector of strings of either "ReLU", "Tanh", or "Sigmoid",
                only for the hidden layers. eg. {"ReLU", "Tanh"} for a net with layer sizes of {2,3,3,2}.
            pUseAsNetwork:
                use it either as a classifier with a one hot output and teaching via labels,
                or as a classical multilayer perceptron with an output vector as desired output.
            pEta:
                step size or teaching rate
            pSeed:
                the seed for the random generator for generating randomized weights
            pUseAdam:
                use Adam optimization or not.
        */
        Network(
            Vec<SIZE> const &pLayerSizes,
            Vec<std::string> const &pActivationFunctionIDs,
            bool const &pUseAsNetwork,
            D const &pEta = .1,
            unsigned int const &pSeed = static_cast<unsigned int>(time(nullptr)),
            bool const &pUseAdam = false
        ) :
        useAsNetwork(pUseAsNetwork),
        eta(pEta),
        step(0),
        i(pLayerSizes[0]) {
            srand(pSeed);
            i.push_back(1.);
            for (SIZE layerID = 1; layerID < pLayerSizes.size(); ++layerID) {
                d.push_back(VD(pLayerSizes[layerID]));
                n.push_back(VD(pLayerSizes[layerID]));
                o.push_back(VD(pLayerSizes[layerID]));
                if (layerID < pLayerSizes.size() - 1) {
                    o[o.size() - 1].push_back(1.);
                }

                D
                wMax = sqrt(6. / static_cast<D>(pLayerSizes[layerID] + pLayerSizes[layerID - 1])),
                wMin = -wMax;
                
                MD
                wTmp = wMin + (wMax - wMin) * mrnd(pLayerSizes[layerID], pLayerSizes[layerID - 1] + 1);
             
                D
                off = .1 * wMax;
                alg::fOrOn(wTmp, [off](D &x){return .9 * x + (x < 0 ? -off : off);});

                w.push_back(wTmp);

                adamV.push_back(mcnst(pLayerSizes[layerID], pLayerSizes[layerID - 1] + 1, 0.));
                adamM.push_back(mcnst(pLayerSizes[layerID], pLayerSizes[layerID - 1] + 1, 0.));

                if (layerID == pLayerSizes.size() - 1) {
                    if (useAsNetwork) {
                        act.push_back(actSoftmax);
                    } else {
                        act.push_back(actSigmoid);
                    }
                } else {
                    if (pActivationFunctionIDs[layerID - 1] == "Sigmoid") {                    
                        act.push_back(actSigmoid);
                        dact.push_back(dActSigmoid);
                    } else if (pActivationFunctionIDs[layerID - 1] == "Tanh") {
                        act.push_back(actTanh);
                        dact.push_back(dActTanh);
                    } else {
                        act.push_back(actReLU);
                        dact.push_back(dActReLU);
                    }
                }
            }
            useAdam = pUseAdam;
            adamBeta1 = .9;
            adamBeta2 = .999;
        }

        ~Network() {

        }

        Network
        & remember() {

            SIZE
            layerID = 0;

            // n[layerID] = w[layerID] | i;
            dot(n[layerID], w[layerID], i);
            act[layerID](n[layerID].cbegin(), n[layerID].cend(), o[layerID].begin());

            while (++layerID < w.size()) {
                // n[layerID] = w[layerID] | o[layerID - 1];
                dot(n[layerID], w[layerID], o[layerID - 1]);
                act[layerID](n[layerID].cbegin(), n[layerID].cend(), o[layerID].begin());
            }

            return *this;
        }

        Network
        & remember(VD const &pInput) {

            presentInput(pInput);

            return remember();
        }

        Network
        & remember(VD::const_iterator pInputBegin) {

            presentInput(pInputBegin);

            return remember();
        }

        Network
        & setActivationFunction(SIZE const &pLayer, std::string const &pActivationFunctionID) {

            if (pLayer < act.size() - 1) {
                if (pActivationFunctionID == "Sigmoid") {
                    act[pLayer]  = actSigmoid;
                    dact[pLayer] = dActSigmoid;
                } else if (pActivationFunctionID == "Tanh") {
                    act[pLayer]  = actTanh;
                    dact[pLayer] = dActTanh;
                } else {
                    act[pLayer]  = actReLU;
                    dact[pLayer] = dActReLU;
                }
            }
            return *this;
        }

        SIZE
        sizeOfInput() const {
            return i.size() - 1;
        }

        SIZE
        outputLayerID() const {
            return o.size() - 1;
        }

        SIZE
        sizeOfOutput() const {
            return o[outputLayerID()].size();
        }

        Network
        & teachLabel(SIZE const &pLabel) {

            ++ step;

            SIZE
            layerID = outputLayerID();

            d[layerID] = o[layerID];
            d[layerID][pLabel] -= 1;
                        
            return teach__(layerID);
        }

        Network
        & teachTarget(VD::const_iterator const &pTargetBegin) {

            ++step;

            if (useAsNetwork) {

                SIZE
                label = static_cast<SIZE>(std::find(pTargetBegin, pTargetBegin + static_cast<ADDR>(sizeOfOutput()), 1.) - pTargetBegin);

                return teachLabel(label);
            }

            SIZE
            layerID = outputLayerID();

            d[layerID] = o[layerID] * (1. - o[layerID]) * (o[layerID] - VD(pTargetBegin, pTargetBegin + static_cast<long int>(o[layerID].size())));
                        
            return teach__(layerID);
        }

        Network
        & teachTarget(VD const &pTarget) {

            if (useAsNetwork) {

                return teachTarget(pTarget.cbegin());
            }

            SIZE
            layerID = outputLayerID();

            d[layerID] = o[layerID] * (1. - o[layerID]) * (o[layerID] - pTarget);
                        
            return teach__(layerID);
        }

        Network
        & teachBatchTargets(VD const &pPatterns, VD const &pTargets) {

            SIZE 
            batchStep = step;

            SIZE
            batchSize = pPatterns.size() / sizeOfInput();

            VD::const_iterator
            patIt = pPatterns.cbegin(),
            tgtIt = pTargets.cbegin();

            for (SIZE batchID = 0; batchID < batchSize; ++batchID) {

                step = batchStep;
                remember(patIt);
                teachTarget(tgtIt);
                patIt += static_cast<ADDR>(sizeOfInput());
                tgtIt += static_cast<ADDR>(sizeOfOutput());
            } 
            step = batchStep + 1;
            
            return *this;
        }

        Network
        & teachBatchLabels(VD const &pPatterns, Vec<SIZE> const &pLabels) {
            SIZE 
            batchStep = step;

            auto
            patIt = pPatterns.cbegin();

            for (auto labIt = pLabels.cbegin(); labIt < pLabels.cend(); ++labIt) {

                step = batchStep;
                remember(patIt);
                teachLabel(*labIt);
                patIt += static_cast<ADDR>(sizeOfInput());
            } 
            step = batchStep + 1;
            return *this;
        }

        VD
        rememberBatchTargets(VD const &pPatterns) {
            SIZE
            numOfPatterns = pPatterns.size() / sizeOfInput();

            VD
            ret(numOfPatterns * sizeOfOutput());

            auto
            tItBeg = ret.begin();

            auto
            cOItBeg = o[outputLayerID()].cbegin(),
            cOItEnd = o[outputLayerID()].cend();

            auto
            cPItBeg = pPatterns.cbegin();

            for (SIZE patternID = 0; patternID < numOfPatterns; ++patternID) {
                remember(cPItBeg);
                cPItBeg += static_cast<ADDR>(sizeOfInput());
                std::copy(cOItBeg, cOItEnd, tItBeg);
                tItBeg += static_cast<ADDR>(sizeOfOutput());
            }             
            return ret;
        }

        Vec<SIZE>
        rememberBatchLabels(VD const &pPatterns) {
            SIZE
            patternSize   = w[0][0].size() - 1,
            numOfPatterns = pPatterns.size() / patternSize;

            Vec<SIZE>
            ret(numOfPatterns);

            VD
            p = VD(patternSize);

            auto
            cOItBeg = o[o.size() - 1].cbegin(),
            cOItEnd = o[o.size() - 1].cend();

            auto
            cPItBeg = pPatterns.cbegin(),
            cPItEnd = pPatterns.cbegin() + static_cast<ADDR>(patternSize);

            for (SIZE patternID = 0; patternID < numOfPatterns; ++patternID) {
                std::copy(cPItBeg, cPItEnd, p.begin());
                remember(p);
                cPItBeg = cPItEnd;
                cPItEnd += static_cast<ADDR>(patternSize);
                ret[patternID] = static_cast<SIZE>(max_element(cOItBeg, cOItEnd) - cOItBeg);
            }             
            return ret;
        }

        static D
        crossEntropy(VD::const_iterator const &pPredictionIter, ADDR const &pLabel) {
            return -log(pPredictionIter[pLabel]);
        }

        static D
        crossEntropy(VD const &pPrediction, ADDR const &pLabel) {
            return crossEntropy(pPrediction.cbegin(), pLabel);
        }

        static D
        crossEntropy(VD const &pPredictions, Vec<SIZE> const &pLabels) {
            ADDR
            sizeOfPrediction = static_cast<ADDR>(pPredictions.size()) / static_cast<ADDR>(pLabels.size());

            VD::const_iterator
            predIt   = pPredictions.cbegin();

            D
            s = 0.;

            for (SIZE sampleID = 0; sampleID < pLabels.size(); ++sampleID) {
                s += crossEntropy(predIt, static_cast<ADDR>(pLabels[sampleID]));
                predIt += sizeOfPrediction;
            }
            return s;
        }

        static D
        rootMeanSquare(VD pPrediction, ADDR const &pLabel) {
            pPrediction[static_cast<SIZE>(pLabel)] -= 1.;
            pPrediction *= pPrediction;            
            return sqrt(sum(pPrediction) / static_cast<double>(pPrediction.size()));
        }

        static D
        rootMeanSquare(VD::const_iterator const &pPredictionBegin, VD::const_iterator const &pPredictionEnd, ADDR const &pLabel) {
            return rootMeanSquare(VD(pPredictionBegin, pPredictionEnd), pLabel);
        }

        static D
        rootMeanSquare(VD const &pPredictions, Vec<SIZE> const &pLabels) {

            ADDR
            sizeOfPrediction = static_cast<ADDR>(pPredictions.size()) / static_cast<ADDR>(pLabels.size());

            VD::const_iterator
            predIt   = pPredictions.cbegin(),
            predItEnd = predIt + static_cast<ADDR>(sizeOfPrediction);

            D
            s = 0.;

            for (SIZE sampleID = 0; sampleID < pLabels.size(); ++sampleID) {

                s += rootMeanSquare(predIt, predItEnd, static_cast<ADDR>(pLabels[sampleID]));
                predIt = predItEnd;
                predItEnd += static_cast<ADDR>(sizeOfPrediction);
            }

            return s;
        }

        static D
        rootMeanSquare(VD const &pPredictions, VD const &pTargets, SIZE const &pBatchSize=1) {

            SIZE
            sizeOfOutput = pPredictions.size() / pBatchSize;

            D
            s = 0.;

            for (SIZE sampleID = 0; sampleID < pBatchSize; ++sampleID) {
                VD
                p(pPredictions.cbegin() + static_cast<ADDR>(sampleID * sizeOfOutput), pPredictions.cbegin() + static_cast<ADDR>((sampleID + 1) * sizeOfOutput)),
                t(pTargets.cbegin() + static_cast<ADDR>(sampleID * sizeOfOutput), pTargets.cbegin() + static_cast<ADDR>((sampleID + 1) * sizeOfOutput)),
                v = p - t;
                s += sqrt(v | v);
            }

            return s;
        }

        // static D
        // crossEntropy(VD const &pPredictions, Vec<SIZE> const &pLabels) {

        //     SIZE
        //     sizeOfPrediction = pPredictions.size() / pLabels.size();

        //     auto
        //     predIt   = pPredictions.cbegin();

        //     D
        //     s = 0.;

        //     for (SIZE sampleID = 0; sampleID < pLabels.size(); ++sampleID) {
        //         s += crossEntropy()
        //     }

        //     while (predIt < pPrediction.cend()) {
        //         auto
        //         predValIt = predIt->cbegin(),
        //         targetValIt = targetIt->cbegin();
        //         while (predValIt < predIt->cend()) {
        //             s += - *targetValIt * log(*predValIt);
        //             ++predValIt;
        //             ++targetValIt;
        //         }        
        //         ++predIt;
        //         ++targetIt;
        //     }

        //     return s;
        // }

        // static D
        // cross_entropy(MD const &pPrediction, MD const &pTarget) {

        //     auto
        //     predIt = pPrediction.cbegin(),
        //     targetIt = pTarget.cbegin();

        //     D
        //     s = 0.;

        //     while (predIt < pPrediction.cend()) {
        //         auto
        //         predValIt = predIt->cbegin(),
        //         targetValIt = targetIt->cbegin();
        //         while (predValIt < predIt->cend()) {
        //             s += - *targetValIt * log(*predValIt);
        //             ++predValIt;
        //             ++targetValIt;
        //         }        
        //         ++predIt;
        //         ++targetIt;
        //     }

        //     return s;
        // }

        // Network
        // & printStatus(MD const &pPatterns, MD const &pTargets, int const &pDigits = 2) {

        //     MD
        //     pred(pPatterns.size());

        //     for (SIZE j = 0; j < pred.size(); ++j) {

        //         pred[j] = remember(pPatterns[j]).o[o.size() - 1];
        //         std::cout << "i[  " << VD(i.cbegin(), i.cend() - 1) << "]   t[  " << pTargets[j] << "]   p[  " << round(pred[j], pDigits) << "]" << std::endl;
        //     }

        //     std::cout << "cross entropy total: " << crossEntropy(pred, pTargets) << std::endl << std::endl;

        //     return *this;
        // }

        // Network
        // & printStatus(VD const &pPatterns, Vec<SIZE> const &pLabels, int const &pDigits = 2) {

        //     SIZE
        //     sizeOfOnePattern = pPatterns.size() / pLabels.size();

        //     auto
        //     patternIt = pPatterns.cbegin();

        //     MD
        //     pred(pLabels.size(), VD(sizeOfOnePattern));

        //     for (SIZE j = 0; j < pred.size(); ++j) {

        //         pred[j] = remember(patternIt).o[o.size() - 1];
        //         std::cout << "i[  " << VD(i.cbegin(), i.cend() - 1) << "]   t[  " << pLabels[j] << "]   p[  " << round(pred[j], pDigits) << "]" << std::endl;
        //         patternIt += sizeOfOnePattern;
        //     }

        //     std::cout << "cross entropy total: " << crossEntropy(pred, pTargets) << std::endl << std::endl;

        //     return *this;
        // }

        VD
        output() const {
            return o[o.size() - 1];
        }
};

#endif // NETWORK_HPP