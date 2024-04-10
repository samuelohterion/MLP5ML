#ifndef TEST3_HPP
#define TEST3_HPP

#include "../AlgebraWithSTL/algebra.hpp"
#include <functional>

using namespace alg;

class Test3 {

    typedef std::function<void(VD::const_iterator, VD::const_iterator, VD::iterator)> ACTIVATIONFUNCTION;
    typedef std::function<void(VD::const_iterator, VD::const_iterator, VD::const_iterator, VD::iterator)> DELTAOUTPUT;
    private:

        Test3
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

             std::transform(pCNetBegin, pCNetEnd, pOutBegin, [](D const &pX){return pX < 0 ? .001 * pX : pX;});
        };

        ACTIVATIONFUNCTION
        dActReLU = [](VD::const_iterator pCOutBegin, VD::const_iterator pCOutEnd, VD::iterator pDstBegin) {

            std::transform(pCOutBegin, pCOutEnd, pDstBegin, [](D const &pX){return pX < 0 ? .001 : 1.;});
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

        DELTAOUTPUT
        deltaOutput;

    public:

        Test3(Vec<SIZE> const &pLayerSizes, D const &pEta) :
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
                D
                wMin = -sqrt(6. / static_cast<D>(pLayerSizes[layerID] + pLayerSizes[layerID - 1])),
                wMax = -wMin;
                w.push_back(wMin + (wMax - wMin) * mrnd(pLayerSizes[layerID], pLayerSizes[layerID - 1] + 1));
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
        }

        ~Test3() {

        }

        Test3
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

        Test3
        & teach(VD const &pTarget) {

            SIZE
            layerID = o.size() - 1;

            dact[layerID](o[layerID].cbegin(), o[layerID].cend(), d[layerID].begin());
            
            //d[layerID] *= (o[layerID] - pTarget);
            assignMultipliedDifference(d[layerID], o[layerID], pTarget);
            
            while (0 < layerID--) {
                dact[layerID](o[layerID].cbegin(), o[layerID].cend() - 1, d[layerID].begin());
                //d[layerID] *= d[layerID + 1] | w[layerID + 1];
                assignMultipliedDotProduct(d[layerID], d[layerID + 1], w[layerID + 1]);
            }
            
            layerID = 0;
            w[layerID] -= eta * d[layerID] ^ i;

            while (++layerID < w.size()) {
                w[layerID] -= eta * d[layerID] ^ o[layerID - 1];
            }
            
            ++step;
            
            return *this;
        }

        Test3
        & teachLabel(SIZE const &pLabel) {

            SIZE
            layerID = o.size() - 1;

            d[layerID] = o[layerID];
            d[layerID][pLabel] -= 1;
            
            while (0 < layerID--) {
                dact[layerID](o[layerID].cbegin(), o[layerID].cend() - 1, d[layerID].begin());
                // d[layerID] *= d[layerID + 1] | w[layerID + 1];
                assignMultipliedDotProduct(d[layerID], d[layerID + 1], w[layerID + 1]);
            }
            
            layerID = 0;
            w[layerID] -= eta * d[layerID] ^ i;

            while (++layerID < w.size()) {
                w[layerID] -= eta * d[layerID] ^ o[layerID - 1];
            }
            
            ++step;
            
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

        Test3
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

#endif // TEST3_HPP