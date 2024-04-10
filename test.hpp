#ifndef TEST_HPP
#define TEST_HPP

#include "../AlgebraWithSTL/algebra.hpp"

using namespace alg;

class Test {

    private:

        Test
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

    public:

        D
        eta,
        wMin,
        wMax;

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

    public:

        Test(Vec<SIZE> const &pLayerSizes, D const &pEta, D const &pWeightsMin, D const &pWeightsMax) :
        eta(pEta),
        wMin(pWeightsMin),
        wMax(pWeightsMax),
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
                w.push_back(wMin + (wMax - wMin) * mrnd(pLayerSizes[layerID], pLayerSizes[layerID - 1] + 1));
            }
        }

        ~Test() {

        }

        Test
        & rem(VD const &pInput) {
            // //////////////////////////////////////////////////////////////////////////////////////////////////////

            // n[0] = w[0] | i;
            // std::transform(n[0].cbegin(), n[0].cend(), o[0].begin(), [](D pN){return 1. / (1. + exp(-pN));});
            
            // n[1] = w[1] | o[0];
            // std::transform(n[1].cbegin(), n[1].cend(), o[1].begin(), [](D pN){return 1. / (1. + exp(-pN));});

            // n[2] = w[2] | o[1];
            // std::transform(n[2].cbegin(), n[2].cend(), o[2].begin(), [](D pN){return 1. / (1. + exp(-pN));});

            // //////////////////////////////////////////////////////////////////////////////////////////////////////

            // auto
            // iInp = pInput.cbegin();
            // auto
            // iIt = i.begin();
            // while (iIt < i.end() - 1) {
            //     *iIt++ = *iInp++;
            // }

            presentInput(pInput);

            SIZE
            layerID = 0;

            n[layerID] = w[layerID] | i;
            std::transform(
                n[layerID].cbegin(), n[layerID].cend(),
                o[layerID].begin(),
                [](D pN){return 1. / (1. + exp(-pN));}
            );

            while (++layerID < w.size()) {
                n[layerID] = w[layerID] | o[layerID - 1];
                std::transform(
                    n[layerID].cbegin(), n[layerID].cend(),
                    o[layerID].begin(),
                    [](D pN){return 1. / (1. + exp(-pN));}
                );
            }

            return *this;
        }

        Test
        & teach(VD const &pTarget) {

            // //////////////////////////////////////////////////////////////////////////////////////////////////////

            // d[2] = (teachers[id] - o[2]);
            
            // std::transform(o[1].cbegin(), o[1].cend() - 1, d[1].begin(), [](D const &pX){return .001 + pX * (1. - pX);});
            // d[1] *= d[2] | w[2];

            // std::transform(o[0].cbegin(), o[0].cend() - 1, d[0].begin(), [](D const &pX){return .001 + pX * (1. - pX);});
            // d[0] *= d[1] | w[1];

            // //////////////////////////////////////////////////////////////////////////////////////////////////////

            SIZE
            layerID = o.size() - 1;

            std::transform(o[layerID].cbegin(), o[layerID].cend(), d[layerID].begin(), [](D const &pX){return .001 + pX * (1. - pX);});
            d[layerID] *= (o[layerID] - pTarget);
            // d[layerID] = pTarget - o[layerID];

            while (0 < layerID--) {
                std::transform(o[layerID].cbegin(), o[layerID].cend() - 1, d[layerID].begin(), [](D const &pX){return .001 + pX * (1. - pX);});
                d[layerID] *= d[layerID + 1] | w[layerID + 1];
            }
            
            //////////////////////////////////////////////////////////////////////////////////////////////////////

            // w[0] -= eta * d[0] ^ i;
            // w[1] -= eta * d[1] ^ o[0];
            // w[2] -= eta * d[2] ^ o[1];
        
            //////////////////////////////////////////////////////////////////////////////////////////////////////
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

        Test
        & printStatus(MD const &pPatterns, MD const &pTargets, int const &pDigits = 2) {

            MD
            pred(pPatterns.size());

            for (SIZE j = 0; j < pred.size(); ++j) {

                pred[j] = rem(pPatterns[j]).o[2];
                std::cout << "i[  " << i << "] -> t[  " << pTargets[j] << "]-->>p[  " << round(pred[j], pDigits) << "]" << std::endl;
            }

            std::cout << "  cs: " << cross_entropy(pred, pTargets) << std::endl << std::endl;

            return *this;
        }
};

#endif // TEST_HPP