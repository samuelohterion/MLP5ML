#include <iostream>
#include <numeric>
#include <random>
// #include "test.hpp"
// #include "test2.hpp"
// #include "test3.hpp"
// #include "test4.hpp"
#include "classifier.hpp"

using namespace alg;

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


int
main() {

    {
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
        loop,
        maxLoops      = 2e2,
        maxLoopsPrint = 1e1;

        unsigned int
        seed = 94891;

        D
        eta  = .1;
        
        Vec<SIZE>
        layerSizes {2, 3, 2};

        bool
        useAdam = true;


        std::cout << "Classifiers --------------------------------------------------------------------------------------------------\n";

        Classifier
        cReLUReLU(layerSizes, {"ReLU",    "ReLU"},    true, eta, seed, useAdam),
        cSigSig(layerSizes,   {"Sigmoid", "Sigmoid"}, true, eta, seed, useAdam),
        cTanhTanh(layerSizes, {"Tanh",    "Tanh"},    true, eta, seed, useAdam);

        // cSigSig.setActivationFunction(0, Classifier::Sigmoid).setActivationFunction(1, Classifier::Sigmoid);
        // cTanhTanh.setActivationFunction(0, Classifier::Tanh).setActivationFunction(1, Classifier::Tanh);
        

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
        
            cReLUReLU.teachBatchLabels(shuffledPatterns, shuffledLabels);
            cSigSig.teachBatchLabels(shuffledPatterns, shuffledLabels);
            cTanhTanh.teachBatchLabels(shuffledPatterns, shuffledLabels);
            
            if ((loop+1) % maxLoopsPrint == 0) {

                VD
                predTargetsReLU = cReLUReLU.rememberBatchTargets(patternsFlat),
                predTargetsSig  = cSigSig.rememberBatchTargets(patternsFlat),
                predTargetsTanh = cTanhTanh.rememberBatchTargets(patternsFlat);

                Vec<SIZE>
                predLabelsReLU = cReLUReLU.rememberBatchLabels(patternsFlat),
                predLabelsSig  = cSigSig.rememberBatchLabels(patternsFlat),
                predLabelsTanh = cTanhTanh.rememberBatchLabels(patternsFlat);

                std::cout << "loop: " << std::setw(1 + static_cast<int>(log10(maxLoops))) << loop + 1 << std::endl
                << "   rmsReLU:    " << Classifier::rootMeanSquare(predTargetsReLU, labels)
                << "   ceReLU:     " << Classifier::crossEntropy(predTargetsReLU, labels)  << std::endl
                << "   rmsSigmoid: " << Classifier::rootMeanSquare(predTargetsSig, labels)
                << "   ceSigmoid:  " << Classifier::crossEntropy(predTargetsSig, labels)  << std::endl
                << "   rmsTanh:    " << Classifier::rootMeanSquare(predTargetsTanh, labels)
                << "   ceTanh:     " << Classifier::crossEntropy(predTargetsTanh, labels)
                << std::endl;
        
                auto
                predItReLU = predTargetsReLU.cbegin(),
                predItSig  = predTargetsSig.cbegin(),
                predItTanh = predTargetsTanh.cbegin();
        
                for (SIZE sampleID = 0; sampleID < labels.size(); ++sampleID) {
                    
                    std::cout 
                    << "i[" << vec2Str(VD(patterns[sampleID].cbegin(), patterns[sampleID].cend()), 2) 
                    << "]   t[  " << targets[sampleID] 
                    << "]   pReLU[" << vec2Str(round(VD(predItReLU, predItReLU + static_cast<long int>(cReLUReLU.sizeOfOutput())), 2), 5)
                    << "]   lReLU[" << predLabelsReLU[sampleID]
                    << "]   pSig["  << vec2Str(round(VD(predItSig, predItSig + static_cast<long int>(cSigSig.sizeOfOutput())), 2), 5)
                    << "]   lSig["  << predLabelsSig[sampleID]
                    << "]   pTanh[" << vec2Str(round(VD(predItTanh, predItTanh + static_cast<long int>(cTanhTanh.sizeOfOutput())), 2), 5)
                    << "]   lTanh[" << predLabelsTanh[sampleID]
                    << "]" << std::endl;
                    predItReLU += static_cast<long int>(cReLUReLU.sizeOfOutput());
                    predItSig  += static_cast<long int>(cSigSig.sizeOfOutput());
                    predItTanh += static_cast<long int>(cTanhTanh.sizeOfOutput());
                }
                std::cout << std::endl;
            }
        }
    }

    {
         MD
        patterns {
            {0., 0.}, 
            {0., 1.}, 
            {1., 0.}, 
            {1., 1.} 
        },
        targets {
            {0.}, 
            {1.}, 
            {1.}, 
            {0.}
        };

        unsigned int
        seed = 1;

        D
        eta  = .1;
        
        bool const
        dontUseAsClassifier = false,
        useAdam = true;

        std::cout << "MLP --------------------------------------------------------------------------------------------\n";

        SIZE
        loop,
        maxLoops      = 2e2,
        maxLoopsPrint = 5e1;

        VD
        patternsFlat = flattenMatrix(patterns),
        targetsFlat  = flattenMatrix(targets);

        Vec<SIZE>
        layerSizes {2, 3, targets[0].size()};

        Classifier
        mlp(layerSizes, {"Tanh"}, dontUseAsClassifier, eta, seed, useAdam);

        Vec<SIZE>
        idx(patterns.size());
        std::iota(idx.begin(), idx.end(), 0);

        for (loop = 0; loop < maxLoops; ++loop) {

            random_shuffle(idx);

            VD
            shuffledPatterns(vcnst(patterns.size() * patterns[0].size())),
            shuffledTargets(vcnst(targets.size() * targets[0].size()));

            for (SIZE batchID = 0; batchID < patterns.size(); ++batchID) {
                for (SIZE i = 0; i < patterns[batchID].size(); ++i) {
                    shuffledPatterns[batchID * patterns[batchID].size() + i] = patterns[idx[batchID]][i];
                }
                for (SIZE i = 0; i < targets[batchID].size(); ++i) {
                    shuffledTargets[batchID * targets[batchID].size() + i] = targets[idx[batchID]][i];
                }
            }
        
            mlp.teachBatchTargets(shuffledPatterns, shuffledTargets);
            
            if ((loop + 1) % maxLoopsPrint == 0) {

                VD
                predTargetsMLP = mlp.rememberBatchTargets(patternsFlat);

                std::cout << "loop: " << std::setw(1 + static_cast<int>(log10(maxLoops))) << loop + 1 << std::endl
                << "   mlp:    " << Classifier::rootMeanSquare(predTargetsMLP, targetsFlat, patterns.size()) << std::endl;
        
                auto
                predItMLP = predTargetsMLP.cbegin();
        
                for (SIZE sampleID = 0; sampleID < patterns.size(); ++sampleID) {
                    
                    std::cout 
                    << "i[" << vec2Str(VD(patterns[sampleID].cbegin(), patterns[sampleID].cend()), 2) 
                    << "]   t[  " << targets[sampleID] 
                    << "]   p[" << vec2Str(round(VD(predItMLP, predItMLP + static_cast<long int>(mlp.sizeOfOutput())), 2), 5)
                    << "]" << std::endl;
                    predItMLP += static_cast<long int>(mlp.sizeOfOutput());
                }
                std::cout << std::endl;
            }
        }
    }


    return 0;
}