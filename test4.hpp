#ifndef TEST4_HPP
#define TEST4_HPP

#include "../AlgebraWithSTL/algebra.hpp"
#include <functional>

using namespace alg;

class Test4 {

    typedef std::function<void(VD::const_iterator, VD::const_iterator, VD::iterator)> ACTIVATIONFUNCTION;
    
    private:

        Test4
        &presentInput(VD const &pInput) {
            auto
            cInpIt = pInput.cbegin();
            auto
            iIt = i.begin();
            while (iIt < i.end() - 1) {
                *iIt++ = *cInpIt++;
            }
            return *this;
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
        assignMultipliedDifference(VD & pC, VD const & pA, VD const & pB) {
            for (SIZE row = 0; row < pC.size(); ++row) {
                pC[row] *= (pA[row] - pB[row]);
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

        Test4
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

        ACTIVATIONFUNCTION
        actSigmoid = [](VD::const_iterator pCNetBegin, VD::const_iterator pCNetEnd, VD::iterator pOutBegin) {

             std::transform(pCNetBegin, pCNetEnd, pOutBegin, [](D const &pX){return 1. / (1. + exp(-pX));});
        };

        ACTIVATIONFUNCTION
        dActSigmoid = [](VD::const_iterator pCOutBegin, VD::const_iterator pCOutEnd, VD::iterator pDstBegin) {

            std::transform(pCOutBegin, pCOutEnd, pDstBegin, [](D const &pX){return pX * (1. - pX);});
        };

        ACTIVATIONFUNCTION
        actTanh = [](VD::const_iterator pCNetBegin, VD::const_iterator pCNetEnd, VD::iterator pOutBegin) {

             std::transform(pCNetBegin, pCNetEnd, pOutBegin, [](D const &pX){return tanh(pX);});
        };

        ACTIVATIONFUNCTION
        dActTanh = [](VD::const_iterator pCOutBegin, VD::const_iterator pCOutEnd, VD::iterator pDstBegin) {

            std::transform(pCOutBegin, pCOutEnd, pDstBegin, [](D const &pX){return 1. - pX * pX;});
        };

        ACTIVATIONFUNCTION
        actReLU = [](VD::const_iterator pCNetBegin, VD::const_iterator pCNetEnd, VD::iterator pOutBegin) {

             std::transform(pCNetBegin, pCNetEnd, pOutBegin, [](D const &pX){return pX < 0 ? .01 * pX : pX;});
        };

        ACTIVATIONFUNCTION
        dActReLU = [](VD::const_iterator pCOutBegin, VD::const_iterator pCOutEnd, VD::iterator pDstBegin) {

            std::transform(pCOutBegin, pCOutEnd, pDstBegin, [](D const &pX){return pX < 0 ? .01 : 1.;});
        };

        ACTIVATIONFUNCTION
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

        ACTIVATIONFUNCTION
        dActSoftmax = [](VD::const_iterator pCOutBegin, VD::const_iterator pCOutEnd, VD::iterator pDstBegin) {

            //std::transform(pCOutBegin, pCOutEnd, pDstBegin, [](D const &pX){return pX < 0 ? .001 : 1.;});
        };


    public:

        D
        eta;

        bool
        useAdam;

    private:

        VD
        i;

        SIZE
        step;

        MD
        d,
        n,
        o;

        TD
        w;

        Vec<ACTIVATIONFUNCTION>
        act,
        dact;

        D
        adamBeta1,
        adamBeta2;

        TD
        adamM,
        adamV;

    public:

        Test4(Vec<SIZE> const &pLayerSizes, D const &pEta = .1, bool const & pUseAdam = false) :
        eta(pEta),
        i(pLayerSizes[0]),
        step(0) {
            i.push_back(1.);
            for (SIZE layerID = 1; layerID < pLayerSizes.size(); ++layerID) {
                d.push_back(VD(pLayerSizes[layerID]));
                n.push_back(VD(pLayerSizes[layerID]));
                o.push_back(VD(pLayerSizes[layerID]));
                if (layerID < pLayerSizes.size() - 1) {
                    o[o.size() - 1].push_back(1.);
                }
                // D
                // wMin = -5. * sqrt(6. / static_cast<D>(pLayerSizes[layerID] + pLayerSizes[layerID - 1])),
                // wMax = -wMin;
                // w.push_back(wMin + (wMax - wMin) * mrnd(pLayerSizes[layerID], pLayerSizes[layerID - 1] + 1));
                D
                wMin = -1. * sqrt(6. / static_cast<D>(pLayerSizes[layerID] + pLayerSizes[layerID - 1])),
                wMax = -wMin;
                
                MD
                wTmp = wMin + (wMax - wMin) * mrnd(pLayerSizes[layerID], pLayerSizes[layerID - 1] + 1);
             
                D
                off = .1;
                alg::fOrOn(wTmp, [off](D &x){return x + (x < 0 ? -off : off);});

                w.push_back(wTmp);

                adamV.push_back(mcnst(pLayerSizes[layerID], pLayerSizes[layerID - 1] + 1, 0.));
                adamM.push_back(mcnst(pLayerSizes[layerID], pLayerSizes[layerID - 1] + 1, 0.));
                if (layerID < pLayerSizes.size() - 1) {
                    act.push_back(actReLU);
                    dact.push_back(dActReLU);
                    // act.push_back(actTanh);
                    // dact.push_back(dActTanh);
                } else {
                    act.push_back(actSoftmax);
                    dact.push_back(dActSoftmax);
                    // act.push_back(actTanh);
                    // dact.push_back(dActTanh);
                }
            }
            useAdam = pUseAdam;
            adamBeta1 = .9;
            adamBeta2 = .999;
        }

        ~Test4() {

        }

        Test4
        & remember(VD const &pInput) {

            presentInput(pInput);

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

        Test4
        & teach(VD const &pTarget) {

            ++step;

            SIZE
            layerID = o.size() - 1;

            dact[layerID](o[layerID].cbegin(), o[layerID].cend(), d[layerID].begin());
            
            //d[layerID] *= (o[layerID] - pTarget);
            assignMultipliedDifference(d[layerID], o[layerID], pTarget);
            
            return teach__(layerID);
        }

        Test4
        & teachLabel(SIZE const &pLabel) {

            ++ step;

            SIZE
            layerID = o.size() - 1;

            d[layerID] = o[layerID];
            d[layerID][pLabel] -= 1;
                        
            return teach__(layerID);
        }

        Test4
        & teachBatch(VD const &pPatterns, VD const &pTargets) {

            SIZE 
            batchStep = step;

            SIZE
            patternSize = i.size() - 1,
            targetSize = o[o.size() - 1].size(),
            batchSize = pTargets.size() / targetSize;

            for (SIZE batchID = 0; batchID < batchSize; ++batchID) {

                step = batchStep;
                remember(VD(pPatterns.cbegin() + batchID * patternSize, pPatterns.cbegin() + (batchID + 1) * patternSize));
                teach(VD(pTargets.cbegin() + batchID * targetSize, pTargets.cbegin() + (batchID + 1) * targetSize));
            } 
            step = batchStep + 1;
            
            return *this;
        }

        Test4
        & teachBatchLabels(VD const &pPatterns, Vec<SIZE> const &pLabels) {

            SIZE 
            batchStep = step;

            SIZE
            batchSize = pLabels.size(),
            patternSize = pPatterns.size() / pLabels.size();

            for (SIZE batchID = 0; batchID < batchSize; ++batchID) {

                step = batchStep;
                remember(VD(pPatterns.cbegin() + batchID * patternSize, pPatterns.cbegin() + (batchID + 1) * patternSize));
                teachLabel(pLabels[batchID]);

            } 
            step = batchStep + 1;

            return *this;
        }

        static D
        cross_entropy(MD const &pPrediction, MD const &pTarget) {

            auto
            predIt = pPrediction.cbegin(),
            targetIt = pTarget.cbegin();

            D
            s = 0.;

            while (predIt < pPrediction.cend()) {
                auto
                predValIt = predIt->cbegin(),
                targetValIt = targetIt->cbegin();
                while (predValIt < predIt->cend()) {
                    s += - *targetValIt * log(*predValIt);
                    ++predValIt;
                    ++targetValIt;
                }        
                ++predIt;
                ++targetIt;
            }

            return s;
        }

        Test4
        & printStatus(MD const &pPatterns, MD const &pTargets, int const &pDigits = 2) {

            MD
            pred(pPatterns.size());

            for (SIZE j = 0; j < pred.size(); ++j) {

                pred[j] = remember(pPatterns[j]).o[o.size() - 1];
                std::cout << "i[  " << VD(i.cbegin(), i.cend() - 1) << "]   t[  " << pTargets[j] << "]   p[  " << round(pred[j], pDigits) << "]" << std::endl;
            }

            std::cout << "cross entropy total: " << cross_entropy(pred, pTargets) << std::endl << std::endl;

            return *this;
        }

        VD
        output() const {
            return o[o.size() - 1];
        }
};

#endif // TEST4_HPP