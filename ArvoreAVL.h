#ifndef ARVOREAVL_H_INCLUDED
#define ARVOREAVL_H_INCLUDED
#include "NoArv.h"
#include <iostream>
#include <fstream>

class ArvoreAVL
{
public:
    ArvoreAVL();
    bool busca(int val);
    void insere(int val);
    void imprime(std::ostream&);
    int calculaAltura(NoArv *p);

private:
    NoArv* raiz; // ponteiro para o No raiz da árvore
    bool auxBusca(NoArv *p, int val);
    NoArv* auxInsere(NoArv *p, int val);
    void imprimePorNivel(NoArv* p, int nivel,std::ostream&);
    NoArv* rotSimplesEsq(NoArv* p);
    NoArv* rotSimplesDir(NoArv* p);
    NoArv* verificaBalanceamento(NoArv *p);
};

#endif // ARVOREAVL_H_INCLUDED
