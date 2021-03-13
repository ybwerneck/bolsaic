#include <iostream>
#include "ArvBinBusca.h"

using namespace std;

ArvBinBusca::ArvBinBusca()
{
    raiz = NULL;
}

bool ArvBinBusca::vazia()
{
    return raiz == NULL;
}

void ArvBinBusca::insere(int estado, int cidade, string ncidade, int x, int y, int C)
{
    NoQArv* val = new NoQArv(estado, cidade, ncidade, x, y, C);
  raiz=auxInsere(raiz,val);
}
void ArvBinBusca::insere(NoQArv* n)
{
    NoQArv* val = new NoQArv(n);
    raiz=auxInsere(raiz, val);
}

NoQArv* ArvBinBusca::auxInsere(NoQArv *p, NoQArv *valu)
{
    if(p == NULL)
    {
        p = valu;

    }
    else if(valu->getCoord().x < p->getCoord().x)
            if(valu->getCoord().y < p->getCoord().y)
                p->setNw(auxInsere(p->getSw(), valu));
            else
                p->setSw(auxInsere(p->getNw(), valu));
    else
            if(valu->getCoord().y < p->getCoord().y)
               p->setNe(auxInsere(p->getNe(), valu));
            else
                p->setSe(auxInsere(p->getSe(), valu));

    return p;
}

NoQArv* ArvBinBusca::busca(coord val)
{
    return auxBusca(raiz, val);
}
NoQArv* ArvBinBusca::busca(int codC) {
    return auxBusca(raiz, codC);

}

NoQArv* ArvBinBusca::auxBusca(NoQArv*p,int codC) {
    if (p == NULL)
        return NULL;

    if (p->getCodC() == codC)
        return p;
    NoQArv* aux;
            aux=auxBusca(p->getSe(), codC);
            if (aux != NULL)
                return aux;
            aux = auxBusca(p->getNe(), codC);
            if (aux != NULL)
                return aux;

            aux = auxBusca(p->getSw(), codC);
            if (aux != NULL)
                return aux;


            aux = auxBusca(p->getNw(), codC);
            if (aux != NULL)
                return aux;
}
NoQArv* ArvBinBusca::auxBusca(NoQArv *p, coord val)
{
    if(p == NULL)
        return NULL;
    else if(p->getCoord() == val)
        return p;
    else if(val.x < p->getCoord().x)
            if(val.y < p->getCoord().y)
                auxBusca(p->getSe(),val);
            else
                auxBusca(p->getNe(),val);
    else
            if(val.y < p->getCoord().y)
                auxBusca(p->getSw(),val);
            else
                auxBusca(p->getNw(),val);

}



void ArvBinBusca::imprime(std::ostream&o)
{
    imprimePorNivel(raiz, 0, o);
}

void ArvBinBusca::imprimePorNivel(NoQArv *p, int nivel, std::ostream& o)
{
    if(p != NULL)
    {
        o << "(" << nivel << ")";
        for(int i = 1; i <= nivel; i++)
            o << "--";
       o << p->toString() << endl;

        if(p->getNe()!=NULL)
        o<<endl<<"NE";
        imprimePorNivel(p->getNe(), nivel+1,o);
        if(p->getNw()!=NULL)
        o<<endl<<"NW";
        imprimePorNivel(p->getNw(), nivel+1,o);
        if(p->getSe()!=NULL)
        o<<endl<<"SE";
        imprimePorNivel(p->getSe(), nivel+1,o);
        if(p->getSw()!=NULL)
        o<<endl<<"SW";
        imprimePorNivel(p->getSw(), nivel+1,o);

    }
}

ArvBinBusca::~ArvBinBusca()
{
//    raiz = libera(raiz);
}

NoQArv* ArvBinBusca::libera(NoQArv *p)
{
    if(p != NULL)
    {
        p->setNe(libera(p->getNe()));
        p->setNw(libera(p->getSe()));
        p->setSw(libera(p->getSe()));
        p->setSe(libera(p->getSe()));
       delete p;
        p = NULL;
    }
    return p;
}

