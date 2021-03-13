#ifndef ARVOREB_H_INCLUDED
#define ARVOREB_H_INCLUDED
#include "NoB.h"

class ArvoreB
{
    private:
    NoB* raiz;
    int ordem;

    public:
    ArvoreB(int ordem);
    void insere(int val);
    bool busca(int val);
    void imprime(ostream &o);
};


#endif // ARVOREB_H_INCLUDED
