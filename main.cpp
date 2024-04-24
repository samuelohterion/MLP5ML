#include <iostream>
#include <numeric>
#include <random>
// #include "test.hpp"
// #include "test2.hpp"
// #include "test3.hpp"
// #include "test4.hpp"
#include "network.hpp"

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
        maxLoops      = 5e2,
        maxLoopsPrint = 1e2;

        unsigned int
        seed = 94891;

        D
        eta  = .1;
        
        Vec<SIZE>
        layerSizes {2, 3, 3, 2};

        bool
        useAdam = true;


        std::cout << "Networks --------------------------------------------------------------------------------------------------\n";

        Network
        cReLUReLU,
        cSigSig,
        cTanhTanh;
        cReLUReLU.config(layerSizes, {"ReLU",    "ReLU"},    true, eta, seed, useAdam);
        cSigSig.config(layerSizes,   {"Sigmoid", "Sigmoid"}, true, eta, seed, useAdam);
        cTanhTanh.config(layerSizes, {"Tanh",    "Tanh"},    true, eta, seed, useAdam);

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
                << "   rmsReLU:    " << Network::rootMeanSquare(predTargetsReLU, labels)
                << "   ceReLU:     " << Network::crossEntropy(predTargetsReLU, labels)  << std::endl
                << "   rmsSigmoid: " << Network::rootMeanSquare(predTargetsSig, labels)
                << "   ceSigmoid:  " << Network::crossEntropy(predTargetsSig, labels)  << std::endl
                << "   rmsTanh:    " << Network::rootMeanSquare(predTargetsTanh, labels)
                << "   ceTanh:     " << Network::crossEntropy(predTargetsTanh, labels)
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

        int
        pTmp = alg::precision;
        alg::precision = 17;
        cReLUReLU.save("ReLU-ReLU-Softmax");
        cSigSig.save("Sigmoid-Sigmoid-Softmax");
        cTanhTanh.save("Tanh-Tanh-Softmax");
        alg::precision = pTmp;
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
        dontUseAsNetwork = false,
        useAdam = true;

        std::cout << "MLP --------------------------------------------------------------------------------------------\n";

        SIZE
        loop,
        maxLoops      = 5e2,
        maxLoopsPrint = 1e2;

        VD
        patternsFlat = flattenMatrix(patterns),
        targetsFlat  = flattenMatrix(targets);

        Vec<SIZE>
        layerSizes {2, 3, targets[0].size()};

        Network
        mlp;
        mlp.config(layerSizes, {"Tanh"}, dontUseAsNetwork, eta, seed, useAdam);

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
                << "   mlp:    " << Network::rootMeanSquare(predTargetsMLP, targetsFlat, patterns.size()) << std::endl;
        
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

            int
            pTmp = alg::precision;
            alg::precision = 17;
            mlp.save("mlp");
            alg::precision = pTmp;
        }

        std::cout << "Load Network --------------------------------------------------------------------------------------\n";

        Network
        nw;

        nw.load("mlp");

        VD
        predTargetsMLP = nw.rememberBatchTargets(patternsFlat);

        std::cout << "loop: " << std::setw(1 + static_cast<int>(log10(maxLoops))) << loop << std::endl
        << "   mlp:    " << Network::rootMeanSquare(predTargetsMLP, targetsFlat, patterns.size()) << std::endl;

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

    return 0;
}