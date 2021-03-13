#ifndef ARVOREB_H_INCLUDED
#define ARVOREB_H_INCLUDED
#include "NoB.h"

class ArvoreB
{
    private:
    NoB* raiz;
    HashTable *t;
    int ordem;

    public:
    ArvoreB(int ordem);
    ArvoreB(int dMin, HashTable *t);
    void insere(int val);
    int busca(int val);
    void imprime(ostream &o);
};


#endif // ARVOREB_H_INCLUDED
