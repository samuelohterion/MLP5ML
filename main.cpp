#include <iostream>
#include <numeric>
#include <random>
// #include "test.hpp"
// #include "test2.hpp"
// #include "test3.hpp"
// #include "test4.hpp"
#include "classifier.hpp"

Vec<SIZE>
random_shuffle(Vec<SIZE> & pVec) {
	
	// std::random_device rd;
	// std::mt19937 g(rd());	

	std::shuffle(pVec.begin(), pVec.end(), std::mt19937(std::random_device{}()));

	return pVec;
}


VD
flattenMatrix(MD const &pMat) {

    VD
    ret(pMat.size() * pMat[0].size());

    auto
    retIt = ret.begin();

    for (auto matRowIt = pMat.cbegin(); matRowIt < pMat.cend(); ++matRowIt) {
        for (auto matColIt = matRowIt->cbegin(); matColIt < matRowIt->cend(); ++matColIt) {
            *retIt++ = *matColIt;
        }
    }

    return ret;
}

// class Classifier {

//     public:

//         D
//         eta,
//         wMin,
//         wMax;

//         MD
//         d,
//         n,
//         o;

//         TD
//         w;

//         VD
//         &actSigmoid(VD &pOut, VD const &pNet) {
//             std::transform(
//                 pNet.cbegin(), pNet.cend(),
//                 pOut.begin(),
//                 [](D const &pX) {return 1. / (1. + exp(-pX));}                
//             );
//             return pOut;
//         }

//         VD
//         &actReLU(VD &pOut, VD const &pNet) {
//             std::transform(
//                 pNet.cbegin(), pNet.cend(),
//                 pOut.begin(),
//                 [](D const &pX) {return pX < 0 ? 0. : pX;}                
//             );
//             return pOut;
//         }

//         void
//         actSoftmax(VD &pOut, VD const &pNet) {
//             D
//             maxNetSum = *std::max_element(pNet.cbegin(), pNet.cend());
//             std::transform(pNet.cbegin(), pNet.cend(), pOut.begin(), [maxNetSum](D const &x){return exp(x - maxNetSum);});
//             D
//             s = 1. / sum(pOut);            
//             for (auto &x : pOut) x *= s;
//         }

//         void
//         dactReLUOfDelta(SIZE const &pLayerID) {
//             D
//             s = 0.;
//             for (SIZE neuronFromID = 0; neuronFromID < len(d[pLayerID]); ++ neuronFromID) {
//                 s = 0.;
//                 for (SIZE neuronToID = 0; neuronToID < len(d[pLayerID + 1]); ++ neuronToID) {
//                     s += d[pLayerID + 1][neuronToID] * w[pLayerID + 1][neuronToID][neuronFromID];
//                 }
//                 d[pLayerID][neuronFromID] = (0 < d[pLayerID][neuronFromID] ? 1 : 0) * s;
//             }
//         }

//         void
//         dactSoftmaxOfCrossEntropy(VD &pDelta, VD const &pOut, SIZE const &pLabel) {
//             auto oIt = pOut.cbegin();
//             auto dIt = pDelta.begin();
//             while (dIt != pDelta.cend()) {
//                 *dIt = *oIt;
//                 ++dIt;
//                 ++oIt;
//             }
//             pDelta[pLabel] -= 1.;
//         }

//         void
//         updateWeights() {
//             for (SIZE layerID = 1; layerID < len(w); ++layerID) {
//                 for (SIZE toID = 0; toID < len(w[layerID]); ++ toID) {
//                     for (SIZE fromID = 0; fromID < len(o[layerID-1]); ++ fromID) {
//                         w[layerID][toID][fromID] += eta * o[layerID-1][fromID] * d[layerID][toID];
//                     }
//                 }
//             }
//         }

//     public:

//         Classifier(Vec<SIZE> const &pLayerSizes, D const &pEta, D const &pWeightsMin, D const &pWeightsMax) :
//         eta(pEta),
//         wMin(pWeightsMin),
//         wMax(pWeightsMax) {
//             D
//             factor = (wMax - wMin);
//             SIZE
//             layerID = 0;
//             d.push_back(VD());
//             n.push_back(VD());
//             o.push_back(VD(pLayerSizes[layerID]));
//             o[layerID].push_back(1.);
//             w.push_back(MD(0,VD(0)));
//             for (++layerID; layerID < len(pLayerSizes) - 1; ++layerID) {
//                 d.push_back(VD(pLayerSizes[layerID]));
//                 n.push_back(VD(pLayerSizes[layerID]));
//                 o.push_back(VD(pLayerSizes[layerID]));
//                 o[layerID].push_back(1.);
//                 w.push_back(wMin + factor * mrnd(pLayerSizes[layerID], pLayerSizes[layerID - 1] + 1));
//             }
//             d.push_back(VD(pLayerSizes[layerID]));
//             n.push_back(VD(pLayerSizes[layerID]));
//             o.push_back(VD(pLayerSizes[layerID]));
//             w.push_back(wMin + factor * mrnd(pLayerSizes[layerID], pLayerSizes[layerID - 1] + 1));
//         }

//         Classifier
//         & remember(VD const &pInput) {

//             auto
//             oIt = o[0].begin();
//             auto
//             ciIt = pInput.cbegin();

//             while(ciIt < pInput.cend()) {
//                 *oIt++ = *ciIt++;
//             }

//             D
//             s;
            
//             SIZE
//             layerID = 1;
            
//             for (; layerID < len(w) - 1; ++layerID) {
//                 for (SIZE toID = 0; toID < len(w[layerID]); ++toID) {
//                     s = 0.;
//                     for (SIZE fromID = 0; fromID < len(w[layerID][toID]); fromID++) {
//                         s += w[layerID][toID][fromID] * o[layerID - 1][fromID];
//                     }
//                     n[layerID][toID] = s;
//                 }
//                 actReLU(o[layerID], n[layerID]);
//             }

//             for (SIZE toID = 0; toID < len(w[layerID]); ++toID) {
//                 s = 0.;
//                 for (SIZE fromID = 0; fromID < len(w[layerID][toID]); fromID++) {
//                     s += w[layerID][toID][fromID] * o[layerID - 1][fromID];
//                 }
//                 n[layerID][toID] = s;
//             }            
//             actSoftmax(o[layerID], n[layerID]);

//             return *this;
//         }

//         Classifier
//         & teach(SIZE const &pLabel) {

//             SIZE
//             layerID = len(w) - 1;

//             dactSoftmaxOfCrossEntropy(d[layerID], o[layerID], pLabel);

//             while(0 < --layerID) {
//                 dactReLUOfDelta(layerID);
//             }

//             updateWeights();

//             return *this;
//         }
// };


using namespace alg;

int
main() {

    MD
    patterns {
        {0., 0.}, 
        {0., 1.}, 
        {1., 0.}, 
        {1., 1.} 
    },
    targets {
        {1., 0.}, 
        {0., 1.}, 
        {0., 1.}, 
        {1., 0.}
    };

    VD
    patternsFlat = flattenMatrix(patterns),
    targetsFlat  = flattenMatrix(targets);

    Vec<SIZE>
    labels {
        0,
        1,
        1,
        0
    };


    SIZE
    maxLoops      = 3e2,
    maxLoopsPrint = 5e1;

    unsigned int
    seed = 81;

    D
    eta  = .1;
    
    Vec<SIZE>
    layerSizes {2, 3, 3, 2};

    bool
    useAdam = true;



    std::cout << "Classifier 0 -------------------------------------------------------------------------------------------------\n";

    Classifier
    classifier0(layerSizes, eta, seed, useAdam);


    SIZE
    loop = 0,
    sampleID = 2;

    for (loop = 0; loop < maxLoops; ++loop) {

        classifier0.teachBatchTargets(patternsFlat, targetsFlat);
        if (loop % maxLoopsPrint == 0) {

            VD
            pred = classifier0.rememberBatchTargets(patternsFlat);

            std::cout << "loop: " << std::setw(1 + static_cast<int>(log10(maxLoops))) << loop
            << "   rms: " << Classifier::rootMeanSquare(pred, labels)
            << "   cs: " << Classifier::crossEntropy(pred, labels)
            << std::endl;
            auto
            predIt = pred.cbegin();
            for (sampleID = 0; sampleID < labels.size(); ++sampleID) {
                
                std::cout << "i[  " << VD(patterns[sampleID].cbegin(), patterns[sampleID].cend()) << "]   t[  " << targets[sampleID] << "]   p[  " << round(VD(predIt, predIt + static_cast<long int>(classifier0.sizeOfOutput())), 2) << "]" << std::endl;
                predIt += static_cast<long int>(classifier0.sizeOfOutput());
            }
            std::cout << std::endl;
        }
    }


    std::cout << "Classifier 1 -------------------------------------------------------------------------------------------------\n";
    
    Classifier
    classifier1(layerSizes, eta, seed, useAdam);

    Vec<SIZE>
    idx(patterns.size());
    std::iota(idx.begin(), idx.end(), 0);

    
    for (loop = 0; loop < maxLoops; ++loop) {

        random_shuffle(idx);

        VD
        shuffledPatterns(vcnst(patterns.size() * patterns[0].size()));

        Vec<SIZE>
        shuffledLabels(labels.size());

        for (SIZE batchID = 0; batchID < patterns.size(); ++batchID) {
            for (SIZE i = 0; i < patterns[batchID].size(); ++i) {
                shuffledPatterns[batchID * patterns[0].size() + i] = patterns[idx[batchID]][i];
            }
            shuffledLabels[batchID] = labels[idx[batchID]];
        }
    
        classifier1.teachBatchLabels(shuffledPatterns, shuffledLabels);
        
        if (loop % maxLoopsPrint == 0) {

            VD
            pred = classifier1.rememberBatchTargets(patternsFlat);

            std::cout << "loop: " << std::setw(1 + static_cast<int>(log10(maxLoops))) << loop
            << "   rms: " << Classifier::rootMeanSquare(pred, labels)
            << "   cs: " << Classifier::crossEntropy(pred, labels)
            << std::endl;
            auto
            predIt = pred.cbegin();
            for (sampleID = 0; sampleID < labels.size(); ++sampleID) {
                
                std::cout << "i[  " << VD(patterns[sampleID].cbegin(), patterns[sampleID].cend()) << "]   t[  " << targets[sampleID] << "]   p[  " << round(VD(predIt, predIt + static_cast<long int>(classifier1.sizeOfOutput())), 2) << "]" << std::endl;
                predIt += static_cast<long int>(classifier1.sizeOfOutput());
            }
            std::cout << std::endl;
        }
    }
 
 
    std::cout << "Classifier 2 -------------------------------------------------------------------------------------------------\n";
    
    Classifier
    classifier2(layerSizes, eta, seed, useAdam);
    
    for (loop = 0; loop < maxLoops; ++loop) {

        random_shuffle(idx);

        VD
        shuffledPatterns(vcnst(patterns.size() * patterns[0].size()));

        Vec<SIZE>
        shuffledLabels(labels.size());

        for (SIZE batchID = 0; batchID < patterns.size(); ++batchID) {
            for (SIZE i = 0; i < patterns[batchID].size(); ++i) {
                shuffledPatterns[batchID * patterns[0].size() + i] = patterns[idx[batchID]][i];
            }
            shuffledLabels[batchID] = labels[idx[batchID]];
        }
    
        classifier2.teachBatchLabels(shuffledPatterns, shuffledLabels);
        
        if (loop % maxLoopsPrint == 0) {

            VD
            pred = classifier2.rememberBatchTargets(patternsFlat);

            Vec<SIZE>
            predLabels = classifier2.rememberBatchLabels(patternsFlat);

            std::cout << "loop: " << std::setw(1 + static_cast<int>(log10(maxLoops))) << loop
            << "   rms: " << Classifier::rootMeanSquare(pred, labels)
            << "   cs: " << Classifier::crossEntropy(pred, labels)
            << std::endl;
            auto
            predIt = pred.cbegin();
            for (sampleID = 0; sampleID < labels.size(); ++sampleID) {
                
                std::cout 
                << "i[  " << VD(patterns[sampleID].cbegin(), patterns[sampleID].cend()) 
                << "]   t[  " << targets[sampleID] 
                << "]   p[  " << round(VD(predIt, predIt + static_cast<long int>(classifier2.sizeOfOutput())), 2)
                << "]   l[  " << predLabels[sampleID]
                 << "  ]" << std::endl;
                predIt += static_cast<long int>(classifier2.sizeOfOutput());
            }
            std::cout << std::endl;
        }
    }
 
    
    std::cout << "Classifiers 3 & 4 --------------------------------------------------------------------------------------------\n";

    Classifier
    classifier3(layerSizes, eta, seed, useAdam),
    classifier4(layerSizes, eta, seed, useAdam);
    // classifier4.act[0] = Classifier::actTanh;
    // classifier4.dact[0] = Classifier::dActTanh;
    
    for (loop = 0; loop < maxLoops; ++loop) {

        random_shuffle(idx);

        VD
        shuffledPatterns(vcnst(patterns.size() * patterns[0].size()));

        Vec<SIZE>
        shuffledLabels(labels.size());

        for (SIZE batchID = 0; batchID < patterns.size(); ++batchID) {
            for (SIZE i = 0; i < patterns[batchID].size(); ++i) {
                shuffledPatterns[batchID * patterns[0].size() + i] = patterns[idx[batchID]][i];
            }
            shuffledLabels[batchID] = labels[idx[batchID]];
        }
    
        classifier3.teachBatchLabels(shuffledPatterns, shuffledLabels);
        classifier4.teachBatchLabels(shuffledPatterns, shuffledLabels);
        
        if (loop % maxLoopsPrint == 0) {

            VD
            predTargets3 = classifier3.rememberBatchTargets(patternsFlat),
            predTargets4 = classifier4.rememberBatchTargets(patternsFlat);

            Vec<SIZE>
            predLabels3 = classifier3.rememberBatchLabels(patternsFlat),
            predLabels4 = classifier4.rememberBatchLabels(patternsFlat);

            std::cout << "loop: " << std::setw(1 + static_cast<int>(log10(maxLoops))) << loop << std::endl
            << "   rms3: " << Classifier::rootMeanSquare(predTargets3, labels)
            << "   cs3: " << Classifier::crossEntropy(predTargets3, labels)  << std::endl
            << "   rms4: " << Classifier::rootMeanSquare(predTargets4, labels)
            << "   cs4: " << Classifier::crossEntropy(predTargets4, labels)
            << std::endl;
    
            auto
            predIt3 = predTargets3.cbegin(),
            predIt4 = predTargets4.cbegin();
    
            for (sampleID = 0; sampleID < labels.size(); ++sampleID) {
                
                std::cout 
                << "i[" << vec2Str(VD(patterns[sampleID].cbegin(), patterns[sampleID].cend()), 2) 
                << "]   t[  " << targets[sampleID] 
                << "]   p3[" << vec2Str(round(VD(predIt3, predIt3 + static_cast<long int>(classifier3.sizeOfOutput())), 2), 6)
                << "]   p4[" << vec2Str(round(VD(predIt4, predIt4 + static_cast<long int>(classifier4.sizeOfOutput())), 2), 6)
                << "]   l3[" << predLabels3[sampleID]
                << "]   l4[" << predLabels4[sampleID]
                << "]" << std::endl;
                predIt3 += static_cast<long int>(classifier3.sizeOfOutput());
                predIt4 += static_cast<long int>(classifier4.sizeOfOutput());
            }
            std::cout << std::endl;
        }
    }
 
    return 0;
}