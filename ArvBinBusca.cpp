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

void ArvBinBusca::insere(int estado, int cidade, string ncidade, double x, double y, int C)
{
    NoQArv* val = new NoQArv(estado, cidade, ncidade, x, y, C);
    auxInsere(val);
}
void ArvBinBusca::insere(NoQArv* n)
{
    NoQArv* val = new NoQArv(n);
    auxInsere( val);
}

void ArvBinBusca::auxInsere(NoQArv *valu)
{
    NoQArv* k = raiz;

    if (raiz == NULL)
        raiz = valu;
    else
    {
        while (k != NULL)
        {
            if (k->getCoord().x < valu->getCoord().x)
                if (k->getCoord().y < valu->getCoord().y)
                    if (k->getSw() != NULL)
                        k = k->getSw();
                    else
                    {
                        k->setSw(valu);
                        break;
                    }
                else if (k->getNw() != NULL)
                    k = k->getNw();
                else
                {
                    k->setNw(valu);

                    break;
                }
            else if (k->getCoord().y < valu->getCoord().y)
                if (k->getSe() != NULL)
                    k = k->getSe();
                else
                {
                    k->setSe(valu);
                    break;
                }
            else if (k->getNe() != NULL)
                k = k->getNe();
            else
            {
                k->setNe(valu);
                break;
            }
        }
    }
}


NoQArv* ArvBinBusca::busca(int codC)
{
    return auxBusca(raiz, codC);

}

NoQArv* ArvBinBusca::auxBusca(NoQArv*p,int codC)
{
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
NoQArv* ArvBinBusca::busca(coord val)
{

    NoQArv* k=raiz;
    while (k != NULL)
    {
        if (k->getCoord().x < val.x)
            if (k->getCoord().y < val.y)
                if (!(k->getCoord() == val))
                    k = k->getSw();
                else
                {
                    return k;
                    
                }
            else if (!(k->getCoord() == val))
                k = k->getNw();
            else
            {
                return k;

                
            }
        else if (k->getCoord().y < val.y)
            if (!(k->getCoord() == val))
                k = k->getSe();
            else
            {
                return k;
                
            }
        else if (!(k->getCoord() ==val))
            k = k->getNe();
        else
        {
            return k;
        }
        return NULL;
    }


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

