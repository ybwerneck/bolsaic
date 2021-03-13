#ifndef ARVBINBUSCA_H_INCLUDED
#define ARVBINBUSCA_H_INCLUDED
#include "NoQArv.h"
#include "Coordenada.h"
#include <iostream>
#include <ostream>
class ArvBinBusca
{
public:
    ArvBinBusca();
    ~ArvBinBusca();
    bool vazia(); // verifica se a árvore está vazia
    NoQArv* busca(coord val);
    void imprime(std::ostream&);
    void insere(int estado, int cidade, std::string ncidade, int x, int y, int C);
    void insere(NoQArv* n);

    NoQArv* busca(int);



private:
    NoQArv* raiz; // ponteiro para o No raiz da árvore
    NoQArv* auxBusca(NoQArv *p, coord val);
    NoQArv* auxBusca(NoQArv* p, int );

    NoQArv* auxInsere(NoQArv *p, NoQArv *val);
    void imprimePorNivel(NoQArv* p, int nivel,std::ostream&o);
    NoQArv* libera(NoQArv *p);
};

#endif // ARVBINBUSCA_H_INCLUDED
