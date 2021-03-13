#ifndef ARVOREAVL_H_INCLUDED
#define ARVOREAVL_H_INCLUDED
#include "NoArv.h"
#include "hash.h"

class ArvoreAVL
{
public:
    ArvoreAVL();
    ArvoreAVL(HashTable *t);
    int busca(int val);
    void insere(int val);
    void imprime(std::ostream&);
    int calculaAltura(NoArv *p);

private:
    HashTable* h;
    NoArv* raiz; // ponteiro para o No raiz da árvore
    int auxBusca(NoArv *p, int val);
    NoArv* auxInsere(NoArv *p, int val);
    void imprimePorNivel(NoArv* p, int nivel,std::ostream&);
    NoArv* rotSimplesEsq(NoArv* p);
    NoArv* rotSimplesDir(NoArv* p);
    NoArv* verificaBalanceamento(NoArv *p);
};

#endif // ARVOREAVL_H_INCLUDED
