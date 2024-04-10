#ifndef AI_HPP

#ifndef ai
#define ai ArtificialIntelligence
#endif // ai

namespace ArtificialIntelligence {

    template< typename T>
    class List {

        public:

            List
            * prev,
            * next;            

            T
            value;

        public:

            List() :
            prev(nullptr),
            next(nullptr) {
                
            }

            ~List() {
                if(prev) {
                    prev->next = next;
                }
                if(next) {
                    next->prev = prev;
                }
                prev = nullptr;
                next = nullptr;
            }

            

            

            // List
            // * popFront(List *pListElement, auto pValue) {
            //     List 
            //     *run = this;

            //     while(run->prev) {
            //         run = run->prev;
            //     }

            //     run->prev = pListElement;
            //     pListElement->next = run;
            //     pListElement->prev = nullptr;
            // }

            // List
            // * pushFront(List *pListElement) {
            //     List 
            //     *run = this;

            //     while(run->prev) {
            //         run = run->prev;
            //     }

            //     run->prev = pListElement;
            //     pListElement->next = run;
            //     pListElement->prev = nullptr;
            // }

            // List
            // * pushBack(List *pListElement) {
            //     List 
            //     *run = this;

            //     while(run->next) {
            //         run = run->next;
            //     }

            //     run->next = pListElement;
            //     pListElement->prev = run;
            //     pListElement->next = nullptr;
            // }

            
    };

    class Perceptron {

        public:

            List<double>
            * neurons;

        public:

            Perceptron(unsigned int pNumberOfNeurons) :
            neurons(nullptr) {
                if (!pNumberOfNeurons) {
                    return;
                }
                List<double>
                * run = new List<double>();
                neurons = run;
                while(--pNumberOfNeurons) {
                    run->next = new List<double>();
                    run->next->prev = run;
                    run = run->next;
                }
            }
            Perceptron
            & allocMemory(unsigned int pNumberOfNeurons = 0) {
                if (!pNumberOfNeurons) {
                    return;
                }
                List<double>
                * run = new List<double>();
                neurons = run;
                while(--pNumberOfNeurons) {
                    run->next = new List<double>();
                    run->next->prev = run;
                    run = run->next;
                }
                run->next = new List<double>();
                run->next->prev = run;
                run = run->next;
            }
    };

}

#define AI_HPP
#endif // AI_HPP